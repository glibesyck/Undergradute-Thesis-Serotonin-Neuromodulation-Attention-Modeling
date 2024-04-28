"""
Includes:
- excitatory & inhibitory neurons of desired quantity;
- receptors kinetics (AMPA, NMDA, GABA_A & GABA_B) of first order.
- external background unspecified random cortical excitatory input through AMPA channel (Possion spike train).
- specific synaptic weights for groups of neurons and channels
- serotonin receptors modelling.
"""

import numpy as np

def network_simulation_run(Ne: int, Ni: int, T: np.ndarray, delta_T: float, start_state: np.ndarray, weights: np.ndarray, spike_trains: np.ndarray, serotonin_weights: np.ndarray, params: dict):
    """
    Inputs:
    - Ne (int): number of excitatory neurons.
    - Ni (int): number of inhibitory neurons.
    - T (np.ndarray of float(s)): array of simlutation time points.
    - delta_T (float): integration constant (time step value).
    - start_state (np.ndarray of float(s))): start values of (v, u) for each neuron in network.
    - weights (np.ndarray of float(s)): synaptic weights (i, j) -> weight from i to j.
    - spike_trains (np.ndarray of zeros and ones): uncorrelated background input through AMPA channel.
    - serotonin_weights (np.ndarray): serotonin weights induced on each of the neuron (which impacts current).
    - params (dict): parameters of neurons DST.

    """
    firings = np.array(np.zeros((len(T), Ne + Ni)))
    states = np.array(np.zeros((len(T), Ne + Ni, 2)))
    conductances = np.array(np.zeros((len(T), Ne + Ni,  4)))
    background_conductance = np.array(np.zeros((len(T), Ne + Ni))) #for each of the neuron only through AMPA channel
    states[0, :, :] = start_state
    synaptic_input = np.array(np.zeros((len(T), Ne + Ni)))
    synaptic_inputs = np.array(np.zeros((len(T), Ne + Ni, 4)))
    background_input = np.array(np.zeros((len(T), Ne + Ni)))

    for i in range(len(T) - 1):

        ## 1️⃣ DECIDE WHO FIRES
        firing_neurons_idx = np.where(states[i, :, 0] >= params["v_peak"])[0] #who is on fire? 🥵
        
        exc_firing_neurons_idx = firing_neurons_idx[firing_neurons_idx < Ne]
        inh_firing_neurons_idx = firing_neurons_idx[firing_neurons_idx >= Ne]

        mask = np.zeros(Ne + Ni, dtype=bool)
        mask[firing_neurons_idx] = True

        exc_mask = np.zeros(Ne + Ni, dtype=bool)
        exc_mask[exc_firing_neurons_idx] = True

        inh_mask = np.zeros(Ne + Ni, dtype=bool)
        inh_mask[inh_firing_neurons_idx] = True

        ## 2️⃣ DECIDE WHO FIRES FROM EXTERNAL INPUT
        external_firing_neurons_idx = np.where(spike_trains[i, :] == 1)[0]
        external_exc_firing_neurons_idx = external_firing_neurons_idx[external_firing_neurons_idx < Ne]
        external_inh_firing_neurons_idx = external_firing_neurons_idx[external_firing_neurons_idx >= Ne]

        external_exc_mask = np.zeros(Ne + Ni, dtype=bool)
        external_exc_mask[external_exc_firing_neurons_idx] = True

        external_inh_mask = np.zeros(Ne + Ni, dtype=bool)
        external_inh_mask[external_inh_firing_neurons_idx] = True

        ## 3️⃣ UPDATE FIRING NEURONS

        firings[i, mask] = 1 #save information who is on fire for raster plot

        ## update states of firing neurons
        states[i, mask, 0] = params["c"][mask] #update of v for firing neurons
        states[i, mask, 1] += params["d"][mask] #update of u for firing neurons

        ## update conductance
        conductances[i, exc_mask, 0] += params["g_ampa"][exc_mask]
        conductances[i, exc_mask, 1] += params["g_nmda"][exc_mask]
        conductances[i, inh_mask, 2] += params["g_gabaa"][inh_mask]
        conductances[i, inh_mask, 3] += params["g_gabab"][inh_mask]

        ## update background conductance
        background_conductance[i, external_exc_mask] += params["g_e_external"][external_exc_mask]
        background_conductance[i, external_inh_mask] += params["g_i_external"][external_inh_mask]

        ## 4️⃣ UPDATE OF SYNAPTIC INPUT
        # AMPA + NMDA + GABA A + GABA B
        synaptic_input[i, :] = - (np.multiply(np.dot(conductances[i, :, 0], weights), states[i, :, 0] - 0 * np.ones((Ne + Ni))) \
        # + np.multiply(np.multiply(np.dot(conductances[i, :, 1], weights), states[i, :, 0] - 0 * np.ones((Ne + Ni))), np.divide(np.divide(np.multiply(states[i, :, 0] + 80 * np.ones((Ne + Ni)), states[i, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))), np.ones(Ne + Ni) + np.divide(np.multiply(states[i, :, 0] + 80 * np.ones((Ne + Ni)), states[i, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))))) \
        + np.multiply(np.multiply(np.dot(conductances[i, :, 1], weights), states[i, :, 0] - 0 * np.ones((Ne + Ni))), np.divide(np.ones((Ne + Ni)), np.ones((Ne + Ni)) + 0.4202*np.exp(0.062 * states[i, :, 0]))) \
        + np.multiply(np.dot(conductances[i, :, 2], weights), states[i, :, 0] + 70 * np.ones((Ne + Ni))) \
        + np.multiply(np.dot(conductances[i, :, 3], weights), states[i, :, 0] + 90 * np.ones((Ne + Ni))))

        synaptic_inputs[i, :, 0] = - (np.multiply(np.dot(conductances[i, :, 0], weights), states[i, :, 0] - 0 * np.ones((Ne + Ni))))
        synaptic_inputs[i, :, 1] = - (np.multiply(np.multiply(np.dot(conductances[i, :, 1], weights), states[i, :, 0] - 0 * np.ones((Ne + Ni))), np.divide(np.ones((Ne + Ni)), np.ones((Ne + Ni)) + 0.4202*np.exp(0.062 * states[i, :, 0]))))
        synaptic_inputs[i, :, 2] = - (np.multiply(np.dot(conductances[i, :, 2], weights), states[i, :, 0] + 70 * np.ones((Ne + Ni))))
        synaptic_inputs[i, :, 3] = - (np.multiply(np.dot(conductances[i, :, 3], weights), states[i, :, 0] + 90 * np.ones((Ne + Ni))))
        

        ## 5️⃣ UPDATE OF EXTERNAL BACKGROUND INPUT
        # only AMPA
        background_input[i, :] = - (np.multiply(background_conductance[i, :], states[i, :, 0] - 0 * np.ones((Ne + Ni))))

        ## 6️⃣ UPDATE OF STATES (U, V)

        states[i + 1, :, 0] = states[i, :, 0] + delta_T*np.divide(np.multiply(params["k"], np.multiply((states[i, :, 0] - params["v_r"]), (states[i, :, 0] - params["v_t"]))) - states[i, :, 1] + np.multiply(serotonin_weights, background_input[i, :] + synaptic_input[i, :]), params["C"])  #update v
        states[i + 1, :, 1] = states[i, :, 1] + delta_T*np.multiply(params["a"], (np.multiply(params["b"], (states[i, :, 0] - params["v_r"]))) - states[i, :, 1]) #update u

        ## 7️⃣ UPDATE OF CONDUCTANCES AFTER UPDATE OF STATES (V, U)

        #for excitatory neurons only AMPA & NMDA should change
        conductances[i + 1, :, 0] = conductances[i, :, 0] - delta_T*np.divide(conductances[i, :, 0], params["tau_ampa"])
        conductances[i + 1, :, 1] = conductances[i, :, 1] - delta_T*np.divide(conductances[i, :, 1], params["tau_nmda"])
        #for inhibitory neurons only GABA (A & B) should change 
        conductances[i + 1, :, 2] = conductances[i, :, 2] - delta_T*np.divide(conductances[i, :, 2], params["tau_gabaa"])
        conductances[i + 1, :, 3] = conductances[i, :, 3] - delta_T*np.divide(conductances[i, :, 3], params["tau_gabab"])

        ## 8️⃣ UPDATE OF BACKGROUND CONDUCTANCE 

        background_conductance[i + 1, :] = background_conductance[i, :] - delta_T*np.divide(background_conductance[i, :], params["tau_ampa"])
        background_conductance[i + 1, :] = background_conductance[i, :] - delta_T*np.divide(background_conductance[i, :], params["tau_ampa"])

    return states, firings, synaptic_input, synaptic_inputs, background_input, conductances, background_conductance