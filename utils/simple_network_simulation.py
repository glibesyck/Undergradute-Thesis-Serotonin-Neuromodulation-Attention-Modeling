"""
Includes:
- excitatory & inhibitory neurons of desired quantity;
- receptors kinetics (AMPA, NMDA, GABA_A & GABA_B) of first order.
Excludes:
- serotonin receptors modelling.
"""

import numpy as np

def network_simulation_run(Ne: int, Ni: int, T: np.ndarray, delta_T: float, start_state: np.ndarray, weights: np.ndarray, currents: np.ndarray, params: dict):
    """
    Inputs:
    - Ne (int): number of excitatory neurons.
    - Ni (int): number of inhibitory neurons.
    - T (np.ndarray of float(s)): array of simlutation time points.
    - delta_T (float): integration constant (time step value).
    - start_state (np.ndarray of float(s))): start values of (v, u) for each neuron in network.
    - weights (np.ndarray of float(s)): synaptic weights (i, j) -> weight from i to j.
    - currents (np.ndarray of float(s) or function): injected current to neuron i at time t.
    - params (dict): parameters of neurons DST.

    """
    firings = np.array(np.zeros((len(T) + 1, Ne + Ni)))
    states = np.array(np.zeros((len(T) + 1, Ne + Ni, 2)))
    conductances = np.array(np.zeros((len(T) + 1, Ne + Ni, Ne + Ni, 4))) #4-dimensional because it depends on synaptic weights unfortunately:( will see what we can do with this!
    states[0, :, :] = start_state
    synaptic_input = np.array(np.zeros((len(T) + 1, Ne + Ni)))
    # times = {
    #     "who_fires": 0,
    #     "masks": 0,
    #     "synaptic_input": 0,
    #     "firing_states": 0,
    #     "non-firing_states": 0,
    #     "firing_conductance": 0,
    #     "non-firing_conductance": 0
    # }

    for i in range(1, len(T)):

        ## DECIDE WHO FIRES
        # start = timeit()

        firing_neurons_idx = np.where(states[i - 1, :, 0] >= params["v_peak"])[0] #who is on fire? 🥵

        # end = timeit()
        # times["who_fires"] += end - start
        # start = timeit()

        exc_firing_neurons_idx = firing_neurons_idx[firing_neurons_idx < Ne]
        inh_firing_neurons_idx = firing_neurons_idx[firing_neurons_idx >= Ne]

        mask = np.zeros(Ne + Ni, dtype=bool)
        mask[firing_neurons_idx] = True

        exc_mask = np.zeros(Ne + Ni, dtype=bool)
        exc_mask[exc_firing_neurons_idx] = True

        exc_inv_mask = np.ones(Ne, dtype=bool)
        exc_inv_mask[exc_firing_neurons_idx] = False
        exc_inv_mask = np.append(exc_inv_mask, np.zeros(Ni, dtype = bool))

        inh_mask = np.zeros(Ne + Ni, dtype=bool)
        inh_mask[inh_firing_neurons_idx] = True

        inh_inv_mask = np.ones(Ne + Ni, dtype=bool)
        inh_inv_mask[inh_firing_neurons_idx] = False
        inh_inv_mask[np.arange(Ne)] = False

        # end = timeit()
        # times["masks"] += end - start
        # start = timeit()


        ## UPDATE OF SYNAPTIC INPUT
        synaptic_input[i, :] = np.multiply(np.sum(conductances[i-1, :, :, 0], axis = 0), states[i-1, :, 0] - 0 * np.ones((Ne + Ni))) \
        + np.multiply(np.multiply(np.sum(conductances[i-1, :, :, 1], axis = 0), states[i-1, :, 0] - 0 * np.ones((Ne + Ni))), np.divide(np.divide(np.multiply(states[i-1, :, 0] + 80 * np.ones((Ne + Ni)), states[i-1, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))), np.ones(Ne + Ni) + np.divide(np.multiply(states[i-1, :, 0] + 80 * np.ones((Ne + Ni)), states[i-1, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))))) \
        + np.multiply(np.sum(conductances[i-1, :, :, 2], axis = 0), states[i-1, :, 0] + 70 * np.ones((Ne + Ni))) \
        + np.multiply(np.sum(conductances[i-1, :, :, 3], axis = 0), states[i-1, :, 0] + 90 * np.ones((Ne + Ni)))
        # AMPA + NMDA + GABA A + GABA B

        # end = timeit()
        # times["synaptic_input"] += end - start
        # start = timeit()

        ## UPDATE OF STATES (U, V)

        ## FOR FIRING NEURONS

        states[i, mask, 0] = params["c"][mask] #update of v for firing neurons
        states[i, mask, 1] = states[i - 1, mask, 1] + params["d"][mask] #update of u for firing neurons
        firings[i, mask] = 1 #save information who is on fire for raster plot

        # end = timeit()
        # times["firing_states"] += end - start
        # start = timeit()


        ## FOR NON-FIRING NEURONS

        states[i, ~mask, 0] = states[i - 1, ~mask, 0] + delta_T*np.divide((np.multiply(params["k"][~mask], np.multiply((states[i - 1, ~mask, 0] - params["v_r"][~mask]), (states[i - 1, ~mask, 0] - params["v_t"][~mask])))) - states[i - 1, ~mask, 1] + currents[i, ~mask] - synaptic_input[i, ~mask], params["C"][~mask])  #update v for non-firing neurons: please observe presence of external input as well as internal (synaptic) one
        states[i, ~mask, 1] = states[i - 1, ~mask, 1] + delta_T*np.multiply(params["a"][~mask], (np.multiply(params["b"][~mask], (states[i - 1, ~mask, 0] - params["v_r"][~mask]))) - states[i - 1, ~mask, 1]) #update u for non-firing neurons

        # end = timeit()
        # times["non-firing_states"] += end - start
        # start = timeit()

        ## UPDATE OF CONDUCTANCES AFTER UPDATE OF (V, U) ##

        ## FOR FIRING NEURONS

        #for excitatory neurons only AMPA & NMDA should change
        conductances[i, exc_mask, :, 0] = conductances[i - 1, exc_mask, :, 0] + weights[exc_mask, :]
        conductances[i, exc_mask, :, 1] = conductances[i - 1, exc_mask, :, 1] + weights[exc_mask, :]

        #for inhibitory neurons only GABA (A & B) should change 
        conductances[i, inh_mask, :, 2] = conductances[i - 1, inh_mask, :, 2] + weights[inh_mask, :]
        conductances[i, inh_mask, :, 3] = conductances[i - 1, inh_mask, :, 3] + weights[inh_mask, :]

        # end = timeit()
        # times["firing_conductance"] += end - start
        # start = timeit()

        ## FOR NON-FIRING NEURONS

        #for excitatory neurons only AMPA & NMDA should change
        conductances[i, exc_inv_mask, :, 0] = conductances[i - 1, exc_inv_mask, :, 0] - delta_T*np.divide(conductances[i - 1, exc_inv_mask, :, 0], params["tau_ampa"][exc_inv_mask])
        conductances[i, exc_inv_mask, :, 1] = conductances[i - 1, exc_inv_mask, :, 1] - delta_T*np.divide(conductances[i - 1, exc_inv_mask, :, 1], params["tau_nmda"][exc_inv_mask])
        #for inhibitory neurons only GABA (A & B) should change 
        conductances[i, inh_inv_mask, :, 2] = conductances[i - 1, inh_inv_mask, :, 2] - delta_T*np.divide(conductances[i - 1, inh_inv_mask, :, 2], params["tau_gabaa"][inh_inv_mask])
        conductances[i, inh_inv_mask, :, 3] = conductances[i - 1, inh_inv_mask, :, 3] - delta_T*np.divide(conductances[i - 1, inh_inv_mask, :, 3], params["tau_gabab"][inh_inv_mask])

        # end = timeit()
        # times["non-firing_conductance"] += end - start

        # print(times)
    
    return states, firings, conductances