# csa_neat_ns
Python implmentation of Neuroevolution of Augmenting Topologies (NEAT) with Objective Search and Novelty Search for colloidal self-assembly toy problem.

env.py --> Environment for colloidal self-assembly toy problem
erl.py --> NEAT implementation
novelty.py --> Novelty search implementation
example_config.ini --> config file with parameters required for NEAT with Objective Search and Novelty Search
utils.py --> helpful functions
run_neat.py --> File used to actually run NEAT with Objective Search and Novelty Search
example folder --> output of running the "run_neat.py" file with the "example_config.ini" config file

# Control Goals
The goal here is to use NEAT with Objective Search and Novelty Search to learn a control policy that adjusts one external input to maximize the colloidal self-assembly order parameter as quickly as possible over a pre-determined batch time. This system is particularly interesting because its dynamics are highly nonlinear and contain multiplicative noise. 

# Help
Please direct all questions to jared.oleary@berkeley.edu
