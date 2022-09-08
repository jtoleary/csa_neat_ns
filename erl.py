###############################################################################
# Do not write bytecode to maintain clean directories
import sys
sys.dont_write_bytecode = True

# Import required packages
import os
import time
import pickle
import numpy as np
import gc
from configparser import ConfigParser
import neat
from neat.config import Config
from neat.reporting import ReporterSet, StdOutReporter, BaseReporter
from neat.population import CompleteExtinctionException
from neat.math_util import mean, stdev
import multiprocessing as mp
import matplotlib.pyplot as plt

from novelty import Novelty
from utils import plot_stats, plot_species, draw_net
###############################################################################

###############################################################################
class NEATConfig(Config):
    
    """
    A container for user-configurable parameters of NEAT (with Novelty Search)
    
    Note 1: This class inherits from neat.config.Config

    Note 2: This class is meant for FeedForward Neural Networks only.
    """
    
    def __init__(self, 
                 ENV,
                 cfg_file,
                 genome_type = neat.DefaultGenome, 
                 reproduction_type = neat.DefaultReproduction, 
                 species_set_type = neat.DefaultSpeciesSet, 
                 stagnation_type = neat.DefaultStagnation):
        
        # Store environment
        self.ENV = ENV
        
        # Get "standard" NEAT information
        Config.__init__(self, 
                        genome_type, 
                        reproduction_type, 
                        species_set_type,
                        stagnation_type, 
                        cfg_file)
        
        # Read config file
        parameters = ConfigParser()
        with open(cfg_file) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)
                
    
        # Get parameters that are not read by "Config" class from
        # neat.config.Config
        EvPar = parameters['Evolve']
        self.use_novelty = eval(EvPar['use_novelty'])
        self.n_gen = int(EvPar['n_gen'])
        self.stoch_reps = int(EvPar['stoch_reps'])
        self.mother_dir = EvPar['mother_dir']
        self.input_names = eval(EvPar['input_names'])
        self.output_names = eval(EvPar['output_names'])
        
        # Put node names in a dictionary format
        self.node_names = {}
        for idx, name in enumerate(self.input_names):
            self.node_names[-idx-1] = name
        for idx, name in enumerate(self.output_names):
            self.node_names[idx] = name
            
        # Initialize novelty
        self.novelty = None
        if self.use_novelty:
            self.novpar = parameters['Novelty']
            self.novelty = Novelty(eval(self.novpar['k']),
                                   eval(self.novpar['threshold']),
                                   eval(self.novpar['limit']),
                                   eval(self.novpar['ndims']))
            
        # For debugging
        self.gen = 0
    
    @staticmethod        
    def eval_genome(genome,
                    config):
        
        """
        Calculates fitness of given neural network
        
        filename --> name of gsd file that stores trajectory information
        
        parameters --> weights and biases of neural network
        
        target --> target lattice, e.g., fcc, bcc, etc.
        """
        # Create neural network
        net = neat.nn.FeedForwardNetwork.create(genome, 
                                                config)
        # Simulate system
        states, inputs = config.ENV.simulate(net)
        
        # Get reward
        reward = config.ENV.calc_reward(states)
        
        # Get behavior
        if config.use_novelty:
            index = int(len(states)/int(config.novpar['ndims']))
            behavior = np.array(states[1::index]).flatten()
        else:
            behavior = states
        
        # Clean up memory
        gc.collect()
        
        return [reward, behavior, states, inputs]
            
###############################################################################    

###############################################################################
class NEATPopulation(object):
    """
    This class implements the core NEAT evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
        
    Note 1: We only support max fitness criterion
    
    Note 2: This supports novelty search
    
    Note 3: This supports FF, CPPN, and CTRNN networks
    
    Note 4: This supports Stochastic NEAT (SNEAT) as well
    """

    def __init__(self, 
                 config, 
                 initial_state=None):
        
        # Store reproduction/stagnation methods
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, 
                                            self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        # Assign fitness criterion
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        
        # Check fitness parameters
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None
        
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)
    
    def parse_fitness(self, fitness, use_novelty):
        
        '''
        Break out fitness tuple into (fitness for selection, actual fitness)
        
        '''
        if use_novelty:
            return self.config.novelty.add(fitness[1]), fitness[0]
        else:
            return fitness[0], fitness[0]

    def run(self,
            fitness_function):
        
        # Check fitness termination/generation inputs
        if self.config.no_fitness_termination and (self.config.n_gen is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        # Initialize tracker
        k = 0
        
        # Initialize list of best genomes
        best_genomes = []
        
        while self.config.n_gen is None or k < self.config.n_gen:
            
            # Increase tracker
            self.config.gen = k
            k += 1

            self.reporters.start_generation(self.generation)
            
            # Evaluate all genomes using the user-provided function.
            genome_list = list(self.population.items())
            for _, gg in genome_list:
                    if gg.fitness == None:
                        gg.fitnesses = []
                        gg.actual_fitnesses = []
            
            # Evaluate all genomes using the user-provided function.
            fitness_function(genome_list, self.config)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                    
                g.fitness_temp, g.actual_fitness_temp = self.parse_fitness(g.fitness, self.config.use_novelty)
                
                # Adjust for stochasticity
                g.fitnesses.append(g.fitness_temp)
                g.actual_fitnesses.append(g.actual_fitness_temp)
                
                g.fitness = np.mean(g.fitnesses)
                g.actual_fitness = np.mean(g.actual_fitnesses)
                            
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.actual_fitness > best.actual_fitness:
                    best = g
            
            best_genomes.append(best)
            
            self.reporters.post_evaluate(self.config, 
                                         self.population, 
                                         self.species, best)
          
            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, 
                                                          self.species,
                                                          self.config.pop_size, 
                                                          self.generation)
            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, 
                                  self.population, 
                                  self.generation)

            self.reporters.end_generation(self.config, 
                                          self.population, 
                                          self.species)
            # Increase generation count
            self.generation += 1


        # Find best-performing genomes (out of winners) and run each 50 times
        top_genomes = list(set(best_genomes))
        
        # Turn list into list of tuples
        for i in range(len(top_genomes)):
            top_genomes[i] = (top_genomes[i].key, top_genomes[i])
        
        # Repeat top genomes to better estimate performance
        for _ in range(self.config.stoch_reps):
            fitness_function(top_genomes, self.config)
            for tg_id, tg in top_genomes:
                __, tg.actual_fitness_temp = self.parse_fitness(tg.fitness, 
                                                                self.config.use_novelty)
                
                tg.actual_fitnesses.append(tg.actual_fitness_temp)
                
        # Assign "actual" fitnesses to top genomes
        for tg_id, tg in top_genomes:
            tg.actual_fitness = np.mean(tg.actual_fitnesses)      
        
        # Create list of top genomes with genome ids
        top_genomes_list = []
        for tg_id, tg in top_genomes:
            top_genomes_list.append(tg)
            
        self.best_genome = sorted(set(top_genomes_list), key=lambda x: x.actual_fitness, reverse=True)[0]
        
        # Account for fitness termination
        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, 
                                          self.generation, 
                                          self.best_genome)
        
        # Clean up memory
        gc.collect()
        
        return self.best_genome
###############################################################################
        
###############################################################################
class NEATSaveReporter(BaseReporter):

    def __init__(self, mother_dir):

        BaseReporter.__init__(self)

        self.best_fitness = -np.inf

        # Make directories for results
        models_dir = os.path.join(mother_dir, 'models')
        visuals_dir = os.path.join(mother_dir, 'visuals')
        runs_dir = os.path.join(mother_dir, 'runs')
        
        self.mkdir(models_dir)
        self.mkdir(visuals_dir)
        self.mkdir(runs_dir)

        # Create CSV file for history and write its header
        self.csvfile = open(runs_dir + '/evolve.csv', 'w')
        self.csvfile.write('Gen,MeanFit,StdFit,MaxFit')
        self.csvfile.write('\n')

    def post_evaluate(self, config, population, species, best_genome):

        fits = [c.actual_fitness for c in population.values()]

        # Save current generation info to history file
        fit_max = max(fits)
        self.csvfile.write('%d,%+5.3f,%+5.3f,%+5.3f' %
                           (config.gen,
                            mean(fits),
                            stdev(fits),
                            fit_max))

        if config.use_novelty:
            novs = [c.fitness for c in population.values()]
            self.csvfile.write(',%+5.3f,%+5.3f,%+5.3f' %
                               (mean(novs), stdev(novs), max(novs)))
        
        
        self.csvfile.write('\n')
        self.csvfile.flush()

    def mkdir(self, name):
        os.makedirs(name, exist_ok=True)
###############################################################################

###############################################################################
class NEATStdOutReporter(StdOutReporter):

    def __init__(self, show_species_detail):

        StdOutReporter.__init__(self, show_species_detail)

    def post_evaluate(self, config, population, species, best_genome):

        # Special report for novelty search
        if config.use_novelty:

            novelties = [c.fitness for c in population.values()]
            nov_mean = mean(novelties)
            nov_std = stdev(novelties)
            best_species_id = species.get_species_id(best_genome.key)
            print('Population\'s average novelty: %3.5f stdev: %3.5f' %
                  (nov_mean, nov_std))
            print('Best novelty: %3.5f - size: (%d,%d) - species %d - id %d' %
                  (best_genome.fitness,
                   best_genome.size()[0],
                   best_genome.size()[1],
                   best_species_id,
                   best_genome.key))
            print('Best actual fitness: %f ' % best_genome.actual_fitness)

        # Ordinary report otherwise
        else:
            StdOutReporter.post_evaluate(self,
                                         config,
                                         population,
                                         species,
                                         best_genome)
###############################################################################

###############################################################################
def NEAT(ENV,
         cfg_file):
    
    # Initialize config file
    config = NEATConfig(ENV,
                        cfg_file)
    
    # Make directories
    os.makedirs(config.mother_dir)
    
    # Track time
    start = time.time()
    
    # Create the population, which is the top-level object for a NEAT run.
    p = NEATPopulation(config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(NEATStdOutReporter(show_species_detail=True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(NEATSaveReporter(config.mother_dir))
    
    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), config.eval_genome)
    
    # Run for up to 300 generations.
    winner = p.run(pe.evaluate)
    
    # Print winner output
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Get total time
    end = time.time()
    print (end-start)
    
    # Draw winning net
    draw_net(config, 
             winner, 
             True, 
             node_names = config.node_names,
             filename=config.mother_dir+'/visuals/genome_structure')
    
    # Plot fitness over time
    plot_stats(stats, 
               ylog = False, 
               view =True,
               filename=config.mother_dir+'/visuals/avg_fitness.svg')
    
    # Plot speciation
    plot_species(stats, 
                 view = True,
                 filename=config.mother_dir+'/visuals/speciation.svg') 
    
    
    # Evaluate actual winner
    winner.states = []
    winner.inputs = []
    for i in range(1000):
        [actual_fitness_temp, 
        _, 
        states_temp, 
        inputs_temp] = config.eval_genome(winner,
                                         config)
        
        winner.actual_fitnesses.append(actual_fitness_temp)
        winner.states.append(states_temp)
        winner.inputs.append(inputs_temp)
        
    # Get final actual fitness
    winner.actual_fitness = np.mean(winner.actual_fitnesses)
    
    # Convert to arrays when necessary
    winner.states = np.vstack(winner.states)
    winner.inputs = np.vstack(winner.inputs)
    
    # Get average states
    winner.states_avg = np.average(winner.states, axis=0)
    winner.inputs_avg = np.average(winner.inputs, axis=0)
    
    # Plot averages
    plt.figure(1)
    plt.plot(winner.states_avg.flatten(), color='r')
    plt.axhline(y=5.0, color='k', ls='--', lw=2)
    plt.xlabel('Time Step', fontsize=20)
    plt.ylabel('C6', fontsize = 20)
    plt.tight_layout()
    plt.savefig(config.mother_dir+'/visuals/states_avg.png')
    
    plt.figure(2)
    plt.plot(winner.inputs_avg.flatten(), color='b')
    plt.xlabel('Time Step', fontsize = 20)
    plt.ylabel('u', fontsize = 20)
    plt.tight_layout()
    plt.savefig(config.mother_dir+'/visuals/inputs_avg.png')
    
    plt.clf()
    
    # Save winning net/genome
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    pickle.dump(net, open(config.mother_dir+'/models/winning_net.dat', 'wb'))
    pickle.dump(winner, open(config.mother_dir+'/models/winner.dat', 'wb'))
        
    return winner
###############################################################################






