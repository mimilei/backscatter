import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import os

"""
THIS FILE IS OLD. 
Produces 4-panel plot of resonance tracking.
Somewhat obsolete as of 4/17/2026. See data_viz_sandbox.py for current standard.
"""

def logmag(x):
    # Convert voltage amplitude ratio to standard decibels (dB)
    return 20 * np.log10(np.abs(x))

def plot_hybrid(data, target_port='s11', save_path=''):
    s11 = data[target_port][:, 0].T
    s11_mag = logmag(s11)
    
    # Extract even frames (assumes two interleaved states based on original logic)
    s11_even = s11_mag[:, ::2]

    # Baseline calibration (defaults to first 11 frames)
    baseline_frames = 11
    if s11_even.shape[1] < baseline_frames:
        baseline_frames = s11_even.shape[1]
        
    s11_cal = np.mean(s11_even[:, :baseline_frames], axis=1).reshape(-1, 1)
    s11_diff = s11_even - s11_cal

    # Dynamically map frequency array based on file parameters
    start_freq = data['start_freq'][0] / 1e6 # Convert to MHz
    stop_freq = data['stop_freq'][0] / 1e6   # Convert to MHz
    freqs = np.linspace(start_freq, stop_freq, s11_diff.shape[0])
    times = np.arange(s11_diff.shape[1])

    # Algorithmic Tracking: Find freq of MAX differential S11 (Modulation Depth)
    res_freq_diff_idx = np.argmax(np.abs(s11_diff), axis=0)
    res_freqs_diff = freqs[res_freq_diff_idx]

    # Smooth the tracking line to remove bin-snapping artifacts
    window_size = 9
    smoothed_res_freqs = np.convolve(res_freqs_diff, np.ones(window_size)/window_size, mode='same')
    
    # Fill edges where convolution is biased
    smoothed_res_freqs[:window_size//2] = smoothed_res_freqs[window_size//2]
    smoothed_res_freqs[-window_size//2:] = smoothed_res_freqs[-window_size//2 - 1]

    # --- Initialize 2x2 Dashboard ---
    fig = plt.figure(figsize=(18, 12))
    name = target_port.upper()

    # Panel 1: Spectrogram overlaid with smoothed resonance track
    ax1 = fig.add_subplot(221)
    pc = ax1.pcolormesh(times, freqs, np.abs(s11_diff), shading='nearest', cmap='inferno')
    ax1.plot(times, smoothed_res_freqs, color='cyan', linewidth=2.5, label='Resonance Track')
    ax1.set_title(f'1. Absolute Differential $|{name}|$ with Tracker')
    ax1.set_xlabel('Time (Frames)')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.legend()
    fig.colorbar(pc, ax=ax1, label='Modulation Depth (|dB|)')

    # Panel 2: Contour map of the raw S11 (Shows the physical antenna dip)
    ax2 = fig.add_subplot(222)
    cp = ax2.contourf(times, freqs, s11_even, levels=25, cmap='viridis')
    ax2.set_title(f'2. Contour Map of Raw ${name}$ Magnitude')
    ax2.set_xlabel('Time (Frames)')
    ax2.set_ylabel('Frequency (MHz)')
    fig.colorbar(cp, ax=ax2, label='Magnitude (dB)')

    # Panel 3: Waterfall Plot of S11 Difference
    ax3 = fig.add_subplot(223, projection='3d')
    T, F = np.meshgrid(times, freqs)
    surf = ax3.plot_surface(T, F, np.abs(s11_diff), cmap='plasma', edgecolor='none')
    ax3.set_title('3. 3D Waterfall of Modulation Depth')
    ax3.set_xlabel('Time (Frames)')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_zlabel('Modulation Depth (|dB|)')
    ax3.view_init(elev=30, azim=-55) # Angle for best view of the trench

    # Panel 4: Amplitude vs Freq Snapshots
    ax4 = fig.add_subplot(224)
    # Get 5 indices spaced evenly across the test duration
    snapshots = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
    colors = plt.cm.copper(np.linspace(0, 1, len(snapshots)))
    
    for i, t in enumerate(snapshots):
        if t < s11_even.shape[1]:
            ax4.plot(freqs, s11_even[:, t], color=colors[i], linewidth=2, label=f'Frame {t}')
        
    ax4.set_title(f'4. Evolution of ${name}$ Response (Snapshots)')
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel(f'Raw ${name}$ Magnitude (dB)')
    ax4.grid(True, alpha=0.4)
    ax4.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", help="Name of the npz file to process", required=True)
    parser.add_argument("--just", action="store_true", help="Enable just one time calibration")
    args = parser.parse_args()

    # Load the data
    try:
        data = np.load(args.filename)
        print("Data loaded successfully.")
        out_base = os.path.dirname(os.path.abspath(args.filename))
        filename_base = os.path.basename(args.filename).replace('.npz', '')
        
        plot_hybrid(data, target_port='s11', save_path=os.path.join(out_base, f"{filename_base}_dashboard_s11.png"))
        plot_hybrid(data, target_port='s21', save_path=os.path.join(out_base, f"{filename_base}_dashboard_s21.png"))
    except Exception as e:
        print(f"Error loading data: {e}")