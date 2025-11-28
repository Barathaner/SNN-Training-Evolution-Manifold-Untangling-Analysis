import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch
import numpy as np
def plot_raster_from_frames(frames, title="Rasterplot nach Transform"):
    """
    Visualisiert einen Rasterplot aus transformierten Spike-Frames.

    frames: numpy array mit Shape [T, 1, 700] (Zeitbins, Kanäle, Sensorgröße)
    title: Titel für die Grafik
    """
    frames = frames.reshape(frames.shape[0], -1)  # Garantiert 2D
    spike_times, neuron_indices = np.nonzero(frames)

    plt.figure(figsize=(10, 5))
    plt.scatter(spike_times, neuron_indices, s=1, marker='|', color='black')
    plt.xlabel("Zeitbins")
    plt.ylabel("Neuron Index")
    plt.title(title)
    plt.show()
    
def plot_raster_from_events(events,title="Rasterplot eines Rohdaten-Spiking Events"):
    # Dynamisch maximale Werte bestimmen
    neurons = events["x"].max() + 1
    timesteps = events["t"].max() + 1

    # Spiketrain-Tensor erzeugen
    spike_tensor = torch.zeros((timesteps, neurons), dtype=torch.float32)

    # Spikes eintragen
    for x, t in zip(events["x"], events["t"]):
        spike_tensor[t, x] = 1

    # Plotten
    fig, ax = plt.subplots(figsize=(12, 6))
    splt.raster(spike_tensor, ax, s=1.5, c="black")
    ax.set_title(title)
    ax.set_xlabel("Zeit (ms)")
    ax.set_ylabel("Neuron")
    plt.show()
    plt.savefig(title.replace(" ", "_") + ".png")


def plot_3d_rate_plot(frames, title="3D Rate Plot nach Transform"):
    vec = frames[:, 0, :]

    # vec: shape (n_time_bins, n_neurons)
    n_time_bins, n_neurons = vec.shape

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(n_neurons)
    _y = np.arange(n_time_bins)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.flatten(), _yy.flatten()
    z = np.zeros_like(x)
    dz = vec.flatten()
    dx = dy = 0.8

    # Farben für die Zeit-Bins
    colors = plt.cm.viridis(np.linspace(0, 1, n_time_bins))
    bar_colors = np.repeat(colors, n_neurons, axis=0)

    ax.bar3d(x, y, z, dx, dy, dz, color=bar_colors, shade=True)
    ax.set_xlabel('Neuron-Index')
    ax.set_ylabel('Zeit-Bin')
    ax.set_zlabel('Feuerrate')
    ax.set_title(title)
    
    # Y-Achse umkehren, damit Zeit-Bin 0 links ist
    ax.invert_yaxis()
    
    plt.tight_layout()
    ax.view_init(elev=40, azim=210)  # elev=Höhe, azim=Azimutwinkel (Drehung um die y-Achse)
    plt.show()
    plt.savefig(title.replace(" ", "_") + ".png")