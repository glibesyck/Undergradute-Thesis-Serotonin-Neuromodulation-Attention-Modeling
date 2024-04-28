import numpy as np #data manipulations
import plotly.subplots as sp #visualizations
import plotly.graph_objs as go #visualizations
from plotly.subplots import make_subplots #visualizations

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


def firing_rates(firings: np.ndarray, seconds: float, delta_T: float, stim_start: list, stim_end: list, title = "Spikes"):
    """Plots all firing times as raster plot."""

    layout = go.Layout(
    xaxis=dict(
        rangeslider=dict(visible=False)
    ),
    yaxis=dict(
        layer="below traces"
    )
    )

    fig = go.Figure(layout=layout)

    firing_indices = np.argwhere(firings == 1)

    trace = go.Scatter(x=firing_indices[:, 0] * delta_T / 1000, y=firing_indices[:, 1],
                           mode='markers', marker=dict(color='black', size=4))
    
    for t_stim_start, t_stim_end in zip(stim_start, stim_end):
        fig.add_vrect(
        x0=t_stim_start,
        x1=t_stim_end,
        fillcolor="white",
        opacity=0.6,
        line_width=0,
        layer="below"
        )

        fig.add_shape(
            type="line",
        x0=t_stim_start,
        x1=t_stim_end,
        y0=-0.15,
        y1=-0.15,
        line=dict(
        color="red",
        width=5,
        ),
        yref="blue",
    )   

    fig.add_trace(trace)

    fig.update_layout(
    xaxis_title='Time (in s)',
    yaxis_title='Neuron',
    title=title,
    )


    return trace, fig

def exc_inh_firing_rates(exc_firings: np.ndarray, inh_firings: np.ndarray, stim_start: list, stim_end: list, delta_T: float, title = "Spikes"):
    """Plots all firing times as raster plot."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Excitatory neurons spikes", "Inhibitory neurons spikes"))


    exc_firing_indices = np.argwhere(exc_firings == 1)
    inh_firing_indices = np.argwhere(inh_firings == 1)
    trace_1 = go.Scatter(x=exc_firing_indices[:, 0] * delta_T / 1000, y=exc_firing_indices[:, 1], mode='markers', marker=dict(color='blue', size=4))
    trace_2 = go.Scatter(x=inh_firing_indices[:, 0] * delta_T / 1000, y=inh_firing_indices[:, 1], mode='markers', marker=dict(color='red', size=4))

    fig.add_trace(trace_1, row=1, col=1)
    fig.add_trace(trace_2, row=2, col=1)

    for t_stim_start, t_stim_end in zip(stim_start, stim_end):
        fig.add_vrect(
        x0=t_stim_start,
        x1=t_stim_end,
        fillcolor="white",
        opacity=0.6,
        line_width=0,
        layer="below"
        )
        fig.add_shape(
        type="line",
        x0=t_stim_start,
        x1=t_stim_end,
        y0=-0.05,
        y1=-0.05,
        line=dict(
        color="blue",
        width=5,
        ),
        yref="paper",
        )   

    fig.update_layout(
    xaxis_title='Time (in s)',
    xaxis2_title='Time (in s)',
    yaxis_title='Neuron',
    title=title,
    yaxis2_title='Neuron',
    )

    return fig

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


