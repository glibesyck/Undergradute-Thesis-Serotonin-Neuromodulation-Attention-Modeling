"""
Includes:
- excitatory & inhibitory neurons of desired quantity;
- receptors kinetics (AMPA, NMDA, GABA_A & GABA_B) of first order.
- external background unspecified random cortical excitatory input through AMPA channel (Possion spike train).
- specific synaptic weights for groups of neurons and channels
- serotonin receptors modelling.
"""

import numpy as np

def network_simulation_run(Ne: int, Ni: int, T: np.ndarray, delta_T: float, start_state: np.ndarray, weights: np.ndarray, currents: np.ndarray, spike_trains: np.ndarray, serotonin_weights: np.ndarray, params: dict):
    """
    Inputs:
    - Ne (int): number of excitatory neurons.
    - Ni (int): number of inhibitory neurons.
    - T (np.ndarray of float(s)): array of simlutation time points.
    - delta_T (float): integration constant (time step value).
    - start_state (np.ndarray of float(s))): start values of (v, u) for each neuron in network.
    - weights (np.ndarray of float(s)): synaptic weights (i, j) -> weight from i to j.
    - currents (np.ndarray of float(s) or function): injected current to neuron i at time t (represents stimulus; if no stimulus is presented, consists only of zeros).
    - spike_trains (np.ndarray of zeros and ones): uncorrelated background input through AMPA channel.
    - serotonin_weights (np.ndarray): serotonin weights induced on each of the neuron (which impacts current).
    - params (dict): parameters of neurons DST.

    """
    firings = np.array(np.zeros((len(T) + 1, Ne + Ni)))
    states = np.array(np.zeros((len(T) + 1, Ne + Ni, 2)))
    #conductance in each time point for each pre-synaptic neuron for each channel (notice that for excitatory pre-synaptic neuron we have: 1)AMPA to exc, 2)NMDA to exc, 3)AMPA to inh, 4)NMDA to inh; same with GABA A and GABA B for inhibitory)
    conductances = np.array(np.zeros((len(T) + 1, Ne + Ni,  4)))
    background_conductance = np.array(np.zeros((len(T) + 1, Ne + Ni))) #for each of the neuron only through AMPA channel
    states[0, :, :] = start_state
    synaptic_input = np.array(np.zeros((len(T) + 1, Ne + Ni)))
    background_input = np.array(np.zeros((len(T) + 1, Ne + Ni)))

    for i in range(1, len(T)):

        ## 1️⃣ DECIDE WHO FIRES
        firing_neurons_idx = np.where(states[i - 1, :, 0] >= params["v_peak"])[0] #who is on fire? 🥵

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

        ## 2️⃣ DECIDE WHO FIRES FROM EXTERNAL INPUT
        external_firing_neurons_idx = np.where(spike_trains[i - 1, :] == 1)[0]
        external_exc_firing_neurons_idx = external_firing_neurons_idx[external_firing_neurons_idx < Ne]
        external_inh_firing_neurons_idx = external_firing_neurons_idx[external_firing_neurons_idx >= Ne]

        external_exc_mask = np.zeros(Ne + Ni, dtype=bool)
        external_exc_mask[external_exc_firing_neurons_idx] = True

        external_exc_inv_mask = np.ones(Ne, dtype=bool)
        external_exc_inv_mask[external_exc_firing_neurons_idx] = False
        external_exc_inv_mask = np.append(external_exc_inv_mask, np.zeros(Ni, dtype = bool))

        external_inh_mask = np.zeros(Ne + Ni, dtype=bool)
        external_inh_mask[external_inh_firing_neurons_idx] = True

        external_inh_inv_mask = np.ones(Ne + Ni, dtype=bool)
        external_inh_inv_mask[external_inh_firing_neurons_idx] = False
        external_inh_inv_mask[np.arange(Ne)] = False

        ## 3️⃣ UPDATE OF SYNAPTIC INPUT
        # AMPA + NMDA + GABA A + GABA B
        synaptic_input[i, :] = - (np.multiply(np.dot(conductances[i-1, :, 0], weights.T), states[i-1, :, 0] - 0 * np.ones((Ne + Ni))) \
        + np.multiply(np.multiply(np.dot(conductances[i-1, :, 1], weights.T), states[i-1, :, 0] - 0 * np.ones((Ne + Ni))), np.divide(np.divide(np.multiply(states[i-1, :, 0] + 80 * np.ones((Ne + Ni)), states[i-1, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))), np.ones(Ne + Ni) + np.divide(np.multiply(states[i-1, :, 0] + 80 * np.ones((Ne + Ni)), states[i-1, :, 0] + 80 * np.ones((Ne + Ni))), np.multiply(60 * np.ones((Ne + Ni)), 60 * np.ones((Ne + Ni)))))) \
        + np.multiply(np.dot(conductances[i-1, :, 2], weights.T), states[i-1, :, 0] + 70 * np.ones((Ne + Ni))) \
        + np.multiply(np.dot(conductances[i-1, :, 3], weights.T), states[i-1, :, 0] + 90 * np.ones((Ne + Ni))))

        ## 4️⃣ UPDATE OF EXTERNAL BACKGROUND INPUT
        # only AMPA
        background_input[i, :] = - (np.multiply(background_conductance[i-1, :], states[i-1, :, 0] - 0 * np.ones((Ne + Ni))))

        ## 5️⃣ UPDATE OF STATES (U, V)

        ## 5️⃣A FOR FIRING NEURONS

        states[i, mask, 0] = params["c"][mask] #update of v for firing neurons
        states[i, mask, 1] = states[i - 1, mask, 1] + params["d"][mask] #update of u for firing neurons
        firings[i, mask] = 1 #save information who is on fire for raster plot

        ## 5️⃣B FOR NON-FIRING NEURONS

        states[i, ~mask, 0] = states[i - 1, ~mask, 0] + delta_T*np.divide((np.multiply(params["k"][~mask], np.multiply((states[i - 1, ~mask, 0] - params["v_r"][~mask]), (states[i - 1, ~mask, 0] - params["v_t"][~mask])))) - states[i - 1, ~mask, 1] + np.multiply(serotonin_weights[~mask], currents[i, ~mask] + background_input[i, ~mask] + synaptic_input[i, ~mask]), params["C"][~mask])  #update v for non-firing neurons: please observe presence of external input as well as internal (synaptic) one
        states[i, ~mask, 1] = states[i - 1, ~mask, 1] + delta_T*np.multiply(params["a"][~mask], (np.multiply(params["b"][~mask], (states[i - 1, ~mask, 0] - params["v_r"][~mask]))) - states[i - 1, ~mask, 1]) #update u for non-firing neurons

        ## 6️⃣ UPDATE OF CONDUCTANCES AFTER UPDATE OF STATES (V, U)

        ## 6️⃣A FOR FIRING NEURONS

        #for excitatory neurons only AMPA & NMDA should change

        #excitatory -> excitatory (notice that both channels have different associated constants)
        conductances[i, exc_mask, 0] = conductances[i - 1, exc_mask, 0] + params["g_ee_ampa"][exc_mask]
        conductances[i, exc_mask, 1] = conductances[i - 1, exc_mask, 1] + params["g_ee_nmda"][exc_mask]

        #excitatory -> inhibitory (notice that both channels have different associated constants)
        conductances[i, exc_mask, 0] = conductances[i - 1, exc_mask, 0] + params["g_ei_ampa"][exc_mask]
        conductances[i, exc_mask, 1] = conductances[i - 1, exc_mask, 1] + params["g_ei_nmda"][exc_mask]

        #for inhibitory neurons only GABA (A & B) should change 

        #inhibitory -> excitatory
        conductances[i, inh_mask, 2] = conductances[i - 1, inh_mask, 2] + params["g_ie_gabaa"][inh_mask]
        conductances[i, inh_mask, 3] = conductances[i - 1, inh_mask, 3] + params["g_ie_gabab"][inh_mask]

        #inhibitory -> excitatory
        conductances[i, inh_mask, 2] = conductances[i - 1, inh_mask, 2] + params["g_ii_gabaa"][inh_mask]
        conductances[i, inh_mask, 3] = conductances[i - 1, inh_mask, 3] + params["g_ii_gabab"][inh_mask]

        ## 6️⃣B FOR NON-FIRING NEURONS

        #for excitatory neurons only AMPA & NMDA should change
        conductances[i, exc_inv_mask, 0] = conductances[i - 1, exc_inv_mask, 0] - delta_T*np.divide(conductances[i - 1, exc_inv_mask, 0], params["tau_ampa"][exc_inv_mask])
        conductances[i, exc_inv_mask, 1] = conductances[i - 1, exc_inv_mask, 1] - delta_T*np.divide(conductances[i - 1, exc_inv_mask, 1], params["tau_nmda"][exc_inv_mask])
        #for inhibitory neurons only GABA (A & B) should change 
        conductances[i, inh_inv_mask, 2] = conductances[i - 1, inh_inv_mask, 2] - delta_T*np.divide(conductances[i - 1, inh_inv_mask, 2], params["tau_gabaa"][inh_inv_mask])
        conductances[i, inh_inv_mask, 3] = conductances[i - 1, inh_inv_mask, 3] - delta_T*np.divide(conductances[i - 1, inh_inv_mask, 3], params["tau_gabab"][inh_inv_mask])

        ## 7️⃣ UPDATE OF BACKGROUND CONDUCTANCE 

        #for excitatory neurons who fire
        background_conductance[i, external_exc_mask] = background_conductance[i - 1, external_exc_mask] + params["g_e_external"][external_exc_mask]

        #for inhibitory neurons who fire
        background_conductance[i, external_inh_mask] = background_conductance[i - 1, external_inh_mask] + params["g_i_external"][external_inh_mask]

        #for excitatory neurons who doesn't fire
        background_conductance[i, external_exc_inv_mask] = background_conductance[i - 1, external_exc_inv_mask] - delta_T*np.divide(background_conductance[i - 1, external_exc_inv_mask], params["tau_ampa"][external_exc_inv_mask])

        #for inhibitory neurons who doesn't fire
        background_conductance[i, external_inh_inv_mask] = background_conductance[i - 1, external_inh_inv_mask] - delta_T*np.divide(background_conductance[i - 1, external_inh_inv_mask], params["tau_ampa"][external_inh_inv_mask])
    
    return states, firings