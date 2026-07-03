import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
import re

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
    filtered = []
    for i in range(window_size, arr.shape[1]):
        dat = arr[:, i]
        window = arr[:, i-window_size:i]
        window = np.mean(window, axis=1)
        filtered.append(dat-window)
    filtered = np.array(filtered).T
    return filtered

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        e.g. "z23a" -> ["z", 23, "a"] for natural sorting """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", help="Directory containing npz files (e.g. data/20260403_test/noise)", required=True)
    parser.add_argument("--just", action="store_true", help="Enable just one time calibration")
    args = parser.parse_args()

    # Find all npz files in the target directory
    search_path = os.path.join(args.dir, "*.npz")
    files = glob.glob(search_path)
    
    # Sort files naturally so data_1, data_2, data_10 come in order
    files.sort(key=alphanum_key)

    if not files:
        print(f"No .npz files found in {args.dir}")
        return

    print(f"Found {len(files)} files in {args.dir}. Concatenating...")

    all_ant1 = []
    all_ant2 = []

    for filename in files:
        print(f"Processing {filename}...")
        data = np.load(filename)
        
        s11 = data['s11'][:, 0].T
        s21 = data['s21'][:, 0].T

        s11_mag = logmag(s11)
        s11_phase = phase(s11)
        s21_mag = logmag(s21)
        s21_phase = phase(s21)

        ant1_s21_mag = s21_mag[:,::2]
        ant2_s21_mag = s21_mag[:,1::2]
        length = np.min((ant1_s21_mag.shape[1], ant2_s21_mag.shape[1]))
        ant1_s21_mag = ant1_s21_mag[:,:length]
        ant2_s21_mag = ant2_s21_mag[:,:length]
        
        if not args.just:
            ant1_s21_mag = subtract_moving_average(ant1_s21_mag, window_size=11)
            ant2_s21_mag = subtract_moving_average(ant2_s21_mag, window_size=11)
        else:
            s11_cal = np.mean(ant1_s21_mag[:,-11:], axis=1).reshape(-1,1)
            s21_cal = np.mean(ant2_s21_mag[:,-11:], axis=1).reshape(-1,1)
            ant1_s21_mag = ant1_s21_mag - s11_cal
            ant2_s21_mag = ant2_s21_mag - s21_cal
            
        all_ant1.append(ant1_s21_mag)
        all_ant2.append(ant2_s21_mag)
        
        # Add a clear visual separator (a dark red vertical line) between files
        separator_width = 3
        separator = np.ones((ant1_s21_mag.shape[0], separator_width)) * 2.0 
        all_ant1.append(separator)
        all_ant2.append(separator)
        
    # Remove the last separator
    all_ant1 = all_ant1[:-1]
    all_ant2 = all_ant2[:-1]

    # Horizontally stack all processed data
    giant_ant1 = np.hstack(all_ant1)
    giant_ant2 = np.hstack(all_ant2)

    import seaborn as sns
    sns.set_theme(style="white", context="talk")

    # Use a wider plot to accommodate the concatenated length
    fig, axs = plt.subplots(2, 1, figsize=(18, 10))
    mode_name = os.path.basename(os.path.normpath(args.dir)).replace('_', ' ').title()

    # Prettier diverging colormap
    cmap = 'vlag'

    # Plot for s11
    sns.heatmap(giant_ant1, ax=axs[0], cmap=cmap, center=0, vmin=-2, vmax=2, cbar_kws={'label': 'Magnitude'})
    axs[0].invert_yaxis() # Put 500 MHz at the bottom
    axs[0].set_title(f'S11 Spectrogram – {mode_name}', pad=15, fontweight='bold')
    axs[0].set_xlabel('Cumulative Time')
    axs[0].set_ylabel('Frequency', labelpad=20)

    # Plot for s21
    sns.heatmap(giant_ant2, ax=axs[1], cmap=cmap, center=0, vmin=-2, vmax=2, cbar_kws={'label': 'Magnitude'})
    axs[1].invert_yaxis() # Put 500 MHz at the bottom
    axs[1].set_title(f'S21 Spectrogram – {mode_name}', pad=15, fontweight='bold')
    axs[1].set_xlabel('Cumulative Time')
    axs[1].set_ylabel('Frequency', labelpad=20)

    # Tick marks matching your frequency spread
    y_ticks = [0, 25, 50, 75, 100]
    y_labels = ['500 MHz', '1.0 GHz', '1.5 GHz', '2.0 GHz', '2.5 GHz']
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels, rotation=0, fontsize=12)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels, rotation=0, fontsize=12)

    # Hide dense x-axis ticks to keep it clean
    axs[0].set_xticks([])
    axs[1].set_xticks([])

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == '__main__':
    main()
