import numpy as np
import torch
from torch.utils.data import Dataset


def single_decision(strength, std=0.03, num_timesteps=75):
    u = np.random.normal(0, std, num_timesteps)
    u[5:46] += strength
    y = np.empty(num_timesteps)
    y.fill(np.sign(strength))
    return u, y
      
def parametric(f_1, f_2, f_min=10, f_max=34, n_time_step=75):
    f_u = lambda f: (f - (f_max+f_min)/2) / (f_max - f_min)
    u = np.zeros(n_time_step)
    u[5:11] = f_u(f_1)
    u[60:71] = f_u(f_2)
    y = np.empty(n_time_step)
    y.fill((f_1-f_2)/(f_max-f_min))
    return u, y



class SingleDecisionDataset(Dataset):
    def __init__(self, n_trials, n_time_step=75):   
        """
        Generate perceptual decision-making data.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        n_time_step : int (Default: 75)
            Number of time steps.
        Returns
        -------
        u : array, shape (n_trials,n_time_step)
            input 
        y : array, shape (n_trials,n_time_step)
            output 
        """
        strength_choices = (3.2/ 100)  * np.array([1,2,4,8,16, -1,-2,-4,-8,-16]) 

        u = np.zeros([n_trials,n_time_step])
        y = np.zeros([n_trials,n_time_step])

        for trial in range(n_trials):
            strength = np.random.choice(strength_choices)
            u[trial], y[trial] = single_decision(strength, num_timesteps=n_time_step)
        
        self.u = torch.tensor(u, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_trials = n_trials
        
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, idx):
        return self.u[idx,:], self.y[idx,:] 



class ParametricDataset(Dataset):
    def __init__(self, n_trials, n_time_step=75):   
        """
        Generate perceptual decision-making data.

        Parameters
        ----------
        n_trials : int
            Number of trials.
        n_time_step : int (Default: 75)
            Number of time steps.
        Returns
        -------
        u : array, shape (n_trials,n_time_step)
            input 
        y : array, shape (n_trials,n_time_step)
            output 
        """
        strength_choices = np.array([10, 14, 18, 22, 26, 30, 34])

        u = np.zeros([n_trials,n_time_step])
        y = np.zeros([n_trials,n_time_step])

        for trial in range(n_trials):
            f1 = np.random.choice(strength_choices)
            f2 = np.random.choice(strength_choices)
            u[trial], y[trial] = parametric(f1, f2, n_time_step=n_time_step)
        
        self.u = torch.tensor(u, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_trials = n_trials
        
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, idx):
        return self.u[idx,:], self.y[idx,:]



