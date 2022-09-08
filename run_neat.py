###############################################################################
# Do not write bytecode to maintain clean directories
import sys
sys.dont_write_bytecode = True

# Import required packages
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from env import sio2_2d
import erl
###############################################################################

###############################################################################
# Instantiate 2D SiO2 system
ENV = sio2_2d()

# Load config file
cfg_file = 'example_config.ini'

winner =  erl.NEAT(ENV, cfg_file)

print (winner.actual_fitness)
###############################################################################