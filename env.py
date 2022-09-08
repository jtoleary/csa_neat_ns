###############################################################################
# Import required packages
import numpy as np
from utils import scale, unscale
###############################################################################

###############################################################################
class sio2_2d(object):
    
    def __init__(self, 
                 dt = 1,
                 ct = 1,
                 bt = 200,
                 x_initial = 0,
                 x_crystalline = 5.0,
                 x_min = 0,
                 x_max = 6,
                 u_min = 0.5,
                 u_max = 4.0):
        
        """
            In-silico representation of collodial self-assembly dynamics as described 
            in:
                
              @inproceedings{tang2013colloidal,
              title={Colloidal self-assembly with model predictive control},
              author={Tang, Xun and Xue, Yuzhen and Grover, Martha A},
              booktitle={2013 American Control Conference},
              pages={4228--4233},
              year={2013},
              organization={IEEE}}
          
            dt --> time step discretization
            bt --> batch time
            x_initial --> initial condition
            x_crystalline --> minimum order parameter value at which system
                              is considered to be crystalline
            x_min/max --> min/max state values
            u_min/max --> min/max input values
        """
        
        # Store system information
        self.dt = dt
        self.ct = ct
        self.bt = bt
        self.N = int(bt/dt) # Number of time steps in episode
        self.Nc = int(ct/dt) # Number of time steps per control action
        self.x_initial = x_initial
        self.x_crystalline = x_crystalline
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
    
    def calc_reward(self,
                    states):
        
        '''
        Function that calculates reward
        
        states --> all recorded states
        
        '''
        
        reward = 7236
        for i in range(len(states)):
            reward -= (states[i]-self.x_max)**2
        return reward
    
    
    def step(self,
             xk, 
             uk):
    
        """
        Moves system forward by one control interval.
        
        xk --> state at time step "k"
        uk --> input at time step "k"
            
        """
        for _ in range(self.Nc):
        
            # Noise
            wk = np.random.randn(1)
            
            # Diffusion coefficient
            D2 = 0.0045*np.exp(-(xk-2.1-0.75*uk)**2)+0.0005
            
            # Drift coefficient
            dFdx = 20*(xk-2.1-0.75*uk)
            dD2dx = -2*(xk-2.1-0.75*uk)*0.0045*np.exp(-(xk-2.1-0.75*uk)**2)
            D1 = -(D2*dFdx-dD2dx)
            
            # Predict forward dynamics
            xk = xk + D1*self.dt + np.sqrt(2*D2*self.dt)*wk
    
        # Return (scaled) float
        return xk
    
    def calc_uk_opt(self,
                    net,
                    xk):
        """
        Calculates optimal input
        
        xk --> state at time step "k"
        net --> neural network that represents control policy
            
        """
        uk = net.activate([scale(xk, self.x_min, self.x_max)])[0]

        return unscale(uk, self.u_min, self.u_max)
    
    
    def simulate(self,
                 net):
        
        states = [self.x_initial]
        inputs = []
        
        for k in range(self.N):
            
            # Get state
            xk = states[k]
            
            # Get "optimal" input (scaled)
            uk = self.calc_uk_opt(net, xk)
            
            # Propagate dynamics
            xk = self.step(xk, uk)
            
            # Update state and input behavior (scaled)
            inputs.append(uk)
            states.append(xk)
        
        return states, inputs
###############################################################################        
    