import numpy as np

def generate_spike_train(Ne: int, Ni: int, v_e: int, v_i: int, T: np.ndarray, delta_T: float):
    """
    Generates spikes which represent cortical external input to the given excitatory and inhibitory neurons in the network. 

    Inputs:
    - Ne (int): number of excitatory neurons.
    - Ni (int): number of inhibitory neurons.
    - v_e (int): firing rate for excitatory neurons.
    - v_i (int): firing rate for inhibitory neurons.
    - T (np.ndarray of float(s)): array of simlutation time points.
    - delta_T (float): integration constant (time step value).
    """
    external_spikes = np.array(np.zeros((len(T) + 1, Ne + Ni))) #1 - spike; 0 - no spike this time!

    #for excitatory
    for index in range(Ne):
        external_spikes[:, index] = np.random.rand(len(T) + 1) < v_e*delta_T
    
    #for inhibitory
    for index in range(Ne, Ne + Ni):
        external_spikes[:, index] = np.random.rand(len(T) + 1) < v_i*delta_T
    
    return external_spikes