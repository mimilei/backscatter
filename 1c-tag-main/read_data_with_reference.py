import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def moving_average(arr, ref_len, win=11):
    arr = np.array(arr)
    kernel = np.ones(win) / win
    arr = np.convolve(arr, kernel, mode="same")
    arr = arr[-ref_len-win:-win]
    idx_start = np.argmin(np.abs(arr+0.12))
    idx_end = np.argmin(np.abs(arr+0.18))
    # arr = arr[idx_start:idx_end]
    arr = arr[idx_start:]
    return arr, idx_start, idx_end


def plot(data, data_ref, args):
    s11 = data['s11'][:, 0].T
    s21 = data['s21'][:, 0].T
    s11_ref = data['s11'][:, 0].T
    s21_ref = data['s21'][:, 0].T

    s11_mag = logmag(s11)
    s11_phase = phase(s11)
    s21_mag = logmag(s21)
    s21_phase = phase(s21)

    s11_mag_ref = logmag(s11_ref)
    s11_phase_ref = phase(s11_ref)
    s21_mag_ref = logmag(s21_ref)
    s21_phase_ref = phase(s21_ref)


    ant1_s21_mag = s21_mag[:,::2]
    ant2_s21_mag = s21_mag[:,1::2]

    ant1_s21_mag_ref = np.mean(s21_mag_ref[:,::2],axis=1)
    ant2_s21_mag_ref = np.mean(s21_mag_ref[:,1::2],axis=1)

    length = np.min((ant1_s21_mag.shape[1],ant2_s21_mag.shape[1]))
    ant1_s21_mag = ant1_s21_mag[:,:length]
    ant2_s21_mag = ant2_s21_mag[:,:length]
    # if not args.just:
    #     ant1_s21_mag = subtract_moving_average(ant1_s21_mag, window_size=11)
    #     ant2_s21_mag = subtract_moving_average(ant2_s21_mag, window_size=11)
    # else:
    #     s11_cal = np.mean(ant1_s21_mag[:,:11],axis=1).reshape(-1,1)
    #     s21_cal = np.mean(ant2_s21_mag[:,:11],axis=1).reshape(-1,1)
    #     # s11_cal = np.mean(ant1_s21_mag[:,-11:],axis=1).reshape(-1,1)
    #     # s21_cal = np.mean(ant2_s21_mag[:,-11:],axis=1).reshape(-1,1)
    #     ant1_s21_mag = ant1_s21_mag - s11_cal
    #     ant2_s21_mag = ant2_s21_mag - s21_cal

    ant1_s21_mag = (ant1_s21_mag.T - ant1_s21_mag_ref.T).T
    ant2_s21_mag = (ant2_s21_mag.T - ant2_s21_mag_ref.T).T


    try:
        label = data['label'][11:length]

        # label = data['label']
        # label, idx_start, idx_end = moving_average(label,ref_len=ant1_s21_mag.shape[1],win=11)
        # ant1_s21_mag = ant1_s21_mag[:, idx_start:]
        # ant2_s21_mag = ant2_s21_mag[:, idx_start:]
        # label = (label - np.min(label)) / (np.max(label) - np.min(label))*100
        # # label = np.clip(label,0,np.max(label))
    except Exception as e:
        print("Error reading label:",e)

    # Create a meshgrid based on the dimensions of s11 (assuming s11 and s21 have the same shape)
    x, y = np.meshgrid(np.arange(ant1_s21_mag.shape[1]), np.arange(ant1_s21_mag.shape[0]))

    # Create two subplots (stacked vertically)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for s11
    pc1 = axs[0].pcolormesh(x, y, ant1_s21_mag, shading='nearest', vmin=-1, vmax=1, cmap='seismic')
    axs[0].set_title('Spectrogram of s11')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    fig.colorbar(pc1, ax=axs[0])

    # Plot for s21
    pc2 = axs[1].pcolormesh(x, y, ant2_s21_mag, shading='nearest', vmin=-1, vmax=1, cmap='seismic')
    # try:
    #     for i, lbl in enumerate(label):
    #         axs[1].text(
    #             i,
    #             0.2,
    #             lbl,
    #             ha='center', va='bottom', color='black', fontsize=10
    #         )
    # except:
    #     pass
    axs[1].set_title('Spectrogram of s21')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    fig.colorbar(pc2, ax=axs[1])

    axs[0].set_yticks([0, 25, 50, 75, 100])
    axs[0].set_yticklabels(['500 MHz', '1.0 GHz', '1.5 GHz', '2.0 GHz', '2.5 GHz'])

    axs[1].set_yticks([0, 25, 50, 75, 100])
    axs[1].set_yticklabels(['500 MHz', '1.0 GHz', '1.5 GHz', '2.0 GHz', '2.5 GHz'])

    try:
        axs[1].plot(label)
    except:
        pass

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", help="Name of the file to process")
    parser.add_argument("-reference", help="Name of the file that is reference state")
    parser.add_argument("--just", action="store_true", help="Enable just one time calibration")
    args = parser.parse_args()

    # Load the data
    filename = args.filename
    data = np.load(filename)

    data_ref = np.load(args.reference)

    print("Data format:")
    print(data.files)

    # for i in range(1,6):
    #     filename = f'data/data_{i}.npz'
    #     data = np.load(filename)
    #     plot(data)
    data = np.load(filename)
    plot(data, data_ref, args)