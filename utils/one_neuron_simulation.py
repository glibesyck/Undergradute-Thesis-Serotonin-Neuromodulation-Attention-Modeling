import numpy as np

def neuron_simulation_run(res_state: np.ndarray, I, T: np.ndarray, delta_T: float, params: dict):
    v_curr, u_curr = res_state
    v = [v_curr]
    u = [u_curr]
    for i in range(len(T)):
        if v_curr >= params["v_peak"]:
            v_new = params["c"]
            u_new = u_curr + params["d"]
        else:
            v_new = v_curr + delta_T*(params["k"]*(v_curr - params["v_r"])*(v_curr - params["v_t"]) - u_curr + I(T[i]))/params["C"]
            u_new = u_curr + delta_T*params["a"]*(params["b"]*(v_curr - params["v_r"]) - u_curr)
        
        v_curr = v_new
        u_curr = u_new
        v.append(v_curr)
        u.append(u_curr)
    
    return v, u