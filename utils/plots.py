import numpy as np #data manipulations
import plotly.subplots as sp #visualizations
import plotly.graph_objs as go #visualizations

#LaTeX workaround
import plotly
from IPython.display import display, HTML
plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))


def plot_membrane_voltage(v: np.array, T: np.array):
    """Plots membrane potential progression over the time."""
    fig = go.Figure()

    trace = go.Scatter(x=T, y=v, mode='lines', name=r"$\text{Membrane voltage} \: (\text{in} \: mV)$")
    fig.add_trace(trace)

    fig.update_layout(title='Membrane voltage',
                      xaxis_title=r'$\text{Time} \: (\text{in} \: ms)$',
                      yaxis_title=r'$mV$')
    
    return trace, fig

def plot_conductance(g: np.array, T: np.array):
    """Plots conductance progression over the time."""
    fig = go.Figure()

    trace = go.Scatter(x=T, y=g, mode='lines', name=r"$\text{Conductance} \: (\text{in} \: \frac{mS}{cm^2})$")
    fig.add_trace(trace)

    fig.update_layout(title='Conductance',
                      xaxis_title=r'$\text{Time} \: (\text{in} \: ms)$',
                      yaxis_title=r'$\frac{mS}{cm^2}$')
    
    return trace, fig

def plot_all_membrane_voltage(v: np.array, T: np.array):
    """Plots all membrane potential progression over the time."""
    fig = go.Figure()

    for i in range(v.shape[1]):
        trace = go.Scatter(x=T, y=v[:, i], mode='lines', name=r"$\text{Membrane voltage for neuron }" f"{i}" r"\: (\text{in} \: mV)$")
        fig.add_trace(trace)

    fig.update_layout(title='Membrane voltage',
                      xaxis_title=r'$\text{Time} \: (\text{in} \: ms)$',
                      yaxis_title=r'$mV$')
    
    return fig


def firing_rates(firings: np.ndarray, T: np.ndarray):
    """Plots all firing times as raster plot."""

    fig = go.Figure()

    firing_indices = np.argwhere(firings == 1)

    trace = go.Scatter(x=firing_indices[:, 0], y=firing_indices[:, 1],
                           mode='markers', marker=dict(color='black', size=4))

    fig.add_trace(trace)

    fig.update_layout(
    xaxis_title='Time Step',
    yaxis_title='Neuron',
    title='Raster Plot',
    yaxis=dict(tickmode='array', tickvals=list(range(firings.shape[1])), ticktext=list(range(1, firings.shape[1] + 1)))
    )


    return trace, fig

def plot_membrane_voltage_against_recovery(voltage: np.array, recovery_variable: np.array):
    """Plots membrane voltage against recovery variable progression over the time forming dynamical plot.
    x-axis - recovery variable (u)
    y-axis - membrane voltage (v)"""
    fig = go.Figure()

    trace = go.Scatter(x=recovery_variable, y=voltage, mode='lines', name='Membrane voltage (in mV) against recovery variable (u)')

    fig.add_trace(go.Scatter(x=recovery_variable, y=voltage, mode='lines', name='Membrane voltage (in mV) against potassium recovery variable (u)'))

    fig.update_layout(title='Membrane voltage against recovery variable',
                      yaxis_title='mV')
    
    return trace, fig

def plot_conductances(g: np.array, T: np.array, labels: list):
    """Plots conductance progressions over the time."""
    fig = go.Figure()

    for index in range(g.shape[1]):

        trace = go.Scatter(x=T, y=g[:, index], mode='lines', name=f"{labels[index]} channel")
        fig.add_trace(trace)

    fig.update_layout(title='Conductance',
                      xaxis_title=r'$\text{Time} \: (\text{in} \: ms)$',
                      yaxis_title=r'$\frac{mS}{cm^2}$')
    
    return fig

def plot_weights(weights: np.array, desired_angle: float, desired_index: int, Ne: int, Ni: int, neuron_type: str):
    """Plots weights distribution for the given preferred orientation of the neuron.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, 360, step = 360 / Ne), y=weights[desired_index, :Ne], mode='lines', name='Weights to Post-Synaptic Excitatory Neuron'))
    fig.add_trace(go.Scatter(x=np.arange(0, 360, step = 360 / Ni), y=weights[desired_index, Ne:], mode='lines', name='Weights to Post-Synaptic Inhibitory Neuron'))

    fig.update_layout(title = f"Weights Distribution for {neuron_type} Pre-Synaptic Neuron of Orientation {desired_angle:.02f}")

    return fig


