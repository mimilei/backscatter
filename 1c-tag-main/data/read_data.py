import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-filename", help="Name of the file to process")
parser.add_argument("--just", action="store_true", help="Enable just one time calibration")
args = parser.parse_args()

# Load the data
filename = args.filename
data = np.load(filename)

print("Data format:")
print(data.files)


def phase(x):
    ymax, ymin = 4, -4
    a = np.angle(x)
    a = np.rad2deg(a)
    try:
        a = a / 360 * abs(ymax-ymin)
        a = a + (ymin+ymax)/2
    except:
        pass
    return a

def logmag(x):
    return 20*np.log10(np.abs(x))

def subtract_moving_average(arr, window_size=11):
    """
    For each row (frequency bin), subtract the moving average computed over the past
    'window_size' time samples (causal filter). For the first few time points, the
    average is computed over the available samples.
    """
    filtered = []
    for i in range(window_size,arr.shape[1]):
        dat = arr[:,i]
        window = arr[:,i-window_size:i]
        window = np.mean(window, axis=1)
        filtered.append(dat-window)
    filtered = np.array(filtered).T
    return filtered


def plot(data):
    s11 = data['s11'][:, 0].T
    s21 = data['s21'][:, 0].T

    s11_mag = logmag(s11)
    s11_phase = phase(s11)
    s21_mag = logmag(s21)
    s21_phase = phase(s21)

    ant1_s21_mag = s21_mag[:,::2]
    ant2_s21_mag = s21_mag[:,1::2]
    length = np.min((ant1_s21_mag.shape[1],ant2_s21_mag.shape[1]))
    ant1_s21_mag = ant1_s21_mag[:,:length]
    ant2_s21_mag = ant2_s21_mag[:,:length]
    if not args.just:
        ant1_s21_mag = subtract_moving_average(ant1_s21_mag, window_size=11)
        ant2_s21_mag = subtract_moving_average(ant2_s21_mag, window_size=11)
    else:
        # s11_cal = np.mean(ant1_s21_mag[:,:11],axis=1).reshape(-1,1)
        # s21_cal = np.mean(ant2_s21_mag[:,:11],axis=1).reshape(-1,1)
        s11_cal = np.mean(ant1_s21_mag[:,-11:],axis=1).reshape(-1,1)
        s21_cal = np.mean(ant2_s21_mag[:,-11:],axis=1).reshape(-1,1)
        ant1_s21_mag = ant1_s21_mag - s11_cal
        ant2_s21_mag = ant2_s21_mag - s21_cal

    # Create a meshgrid based on the dimensions of s11 (assuming s11 and s21 have the same shape)
    x, y = np.meshgrid(np.arange(ant1_s21_mag.shape[1]), np.arange(ant1_s21_mag.shape[0]))

    # Create two subplots (stacked vertically)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for s11
    pc1 = axs[0].pcolormesh(x, y, ant1_s21_mag, shading='nearest', vmin=-2, vmax=2, cmap='seismic')
    axs[0].set_title('Spectrogram of s11')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    fig.colorbar(pc1, ax=axs[0])

    # Plot for s21
    pc2 = axs[1].pcolormesh(x, y, ant2_s21_mag, shading='nearest', vmin=-2, vmax=2, cmap='seismic')
    axs[1].set_title('Spectrogram of s21')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    fig.colorbar(pc2, ax=axs[1])

    axs[0].set_yticks([0, 25, 50, 75, 100])
    axs[0].set_yticklabels(['500 MHz', '1.0 GHz', '1.5 GHz', '2.0 GHz', '2.5 GHz'])

    axs[1].set_yticks([0, 25, 50, 75, 100])
    axs[1].set_yticklabels(['500 MHz', '1.0 GHz', '1.5 GHz', '2.0 GHz', '2.5 GHz'])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # for i in range(1,6):
    #     filename = f'data/data_{i}.npz'
    #     data = np.load(filename)
    #     plot(data)
    data = np.load(filename)
    plot(data)