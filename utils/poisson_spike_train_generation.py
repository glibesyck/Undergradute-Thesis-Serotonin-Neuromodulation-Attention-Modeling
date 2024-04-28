import numpy as np

def generate_spike_train(Ne: int, Ni: int, v: np.ndarray, T: np.ndarray, delta_T: float):
    """
    Generates spikes which represent cortical external input and stimuli to the given excitatory and inhibitory neurons in the network. 

    Inputs:
    - Ne (int): number of excitatory neurons.
    - Ni (int): number of inhibitory neurons.
    - v (np.ndarray): firing rates for excitatory and inhibitory neurons through time.
    - T (np.ndarray of float(s)): array of simlutation time points.
    - delta_T (float): integration constant (time step value).
    """
    external_spikes = np.array(np.zeros((len(T), Ne + Ni))) #1 - spike; 0 - no spike this time!
    external_spikes = np.random.rand(len(T), Ne + Ni) < v * delta_T
    return external_spikes