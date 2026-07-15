import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.covariance import LedoitWolf

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
    # Clamp away from exact zero: a zero-magnitude reading (dropped sample/glitch)
    # would otherwise produce -inf and crash any downstream finite-value check
    # (e.g. sklearn's Mahalanobis validation), rather than a very-low-but-finite dB value.
    return 20*np.log10(np.maximum(np.abs(x), 1e-12))

def phase_components(x):
    """Wraparound-safe phase representation: (cos, sin) of the complex angle.

    Raw phase in degrees jumps discontinuously across the +-180 degree boundary, which
    would inject spurious spikes into subtract_moving_average(). cos/sin are bounded in
    [-1, 1] with no discontinuity, so they compose safely with the same moving-average
    filter used for magnitude.
    """
    angle = np.angle(x)
    return np.cos(angle), np.sin(angle)

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

def fit_covariance_reference(features):
    """Fit a shrinkage-regularized (Ledoit-Wolf) covariance estimator on the given
    feature rows. Single place a covariance/precision estimator gets fit in this
    codebase -- fit_noise_reference() is just this applied to noise-only rows.

    features: float64 array, shape (n_rows, n_features). Each row is one
    already-windowed sample -- i.e. already put through logmag + subtract_moving_average
    (or the streaming equivalent) upstream. This function does no windowing itself.
    """
    return LedoitWolf().fit(features)

def fit_noise_reference(noise_features):
    """Fit a shrinkage-regularized covariance model on baseline noise feature vectors."""
    return fit_covariance_reference(noise_features)

def mahalanobis_distance(features, noise_reference):
    """Mahalanobis distance of each row in `features` (same already-windowed representation
    as noise_features above) from the fitted noise reference."""
    return np.sqrt(noise_reference.mahalanobis(features))

def pairwise_mahalanobis(a, b, precision):
    """Mahalanobis distance between two arbitrary feature vectors (e.g. two class
    centroids, or two file centroids) under a shared covariance metric. Unlike
    mahalanobis_distance() (distance of rows to a reference's OWN fitted mean), this
    takes the estimator's `.precision_` directly: sqrt((a-b) @ precision @ (a-b))."""
    diff = np.asarray(a) - np.asarray(b)
    return np.sqrt(diff @ precision @ diff)

def rolling_std(arr, window_size=11):
    """
    For each row (frequency bin), compute the standard deviation over the past
    'window_size' time samples (causal). Captures short-term signal variability
    (e.g. actively flowing liquid perturbing the resonance) as distinct from
    subtract_moving_average's single-instant deviation from the recent mean.
    """
    filtered = []
    for i in range(window_size, arr.shape[1]):
        window = arr[:, i-window_size:i]
        filtered.append(np.std(window, axis=1))
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


def plot(data, args):
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
        s11_cal = np.mean(ant1_s21_mag[:,:11],axis=1).reshape(-1,1)
        s21_cal = np.mean(ant2_s21_mag[:,:11],axis=1).reshape(-1,1)
        # s11_cal = np.mean(ant1_s21_mag[:,-11:],axis=1).reshape(-1,1)
        # s21_cal = np.mean(ant2_s21_mag[:,-11:],axis=1).reshape(-1,1)
        ant1_s21_mag = ant1_s21_mag - s11_cal
        ant2_s21_mag = ant2_s21_mag - s21_cal


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
    pc1 = axs[0].pcolormesh(x, y, ant1_s21_mag, shading='nearest', vmin=-3, vmax=3, cmap='seismic')
    axs[0].set_title('Spectrogram of s11')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    fig.colorbar(pc1, ax=axs[0])

    # Plot for s21
    pc2 = axs[1].pcolormesh(x, y, ant2_s21_mag, shading='nearest', vmin=-3, vmax=3, cmap='seismic')
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
    parser.add_argument("--just", action="store_true", help="Enable just one time calibration")
    args = parser.parse_args()

    # Load the data
    filename = args.filename
    data = np.load(filename)

    print("Data format:")
    print(data.files)

    # for i in range(1,6):
    #     filename = f'data/data_{i}.npz'
    #     data = np.load(filename)
    #     plot(data)
    data = np.load(filename)
    plot(data, args)