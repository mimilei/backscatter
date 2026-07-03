import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import skrf as rf
import os
import argparse
from scipy.ndimage import gaussian_filter
import argparse
from scipy.ndimage import gaussian_filter

"""
This is the most updated data viz file. 

Example: 
python data_viz_sandbox.py -filename data/20260424_nacl/trial3_2ml/20260424_210039_1.npz --just --viz_keypress

python data_viz_sandbox.py -filename path/to/file.npz --prefix "1mL/min"
"""

# Global Seaborn Theme Settings
sns.set_theme(style="whitegrid", context="talk")


def logmag(x):
    return 20 * np.log10(np.abs(x))

def smooth_track(track, target_window=9):
    window_size = min(target_window, len(track))
    if window_size % 2 == 0 and window_size > 0:
        window_size -= 1
    if window_size < 1:
        window_size = 1
    
    smoothed = np.convolve(track, np.ones(window_size)/window_size, mode='same')
    if window_size > 1:
        smoothed[:window_size//2] = smoothed[window_size//2]
        smoothed[-window_size//2:] = smoothed[-window_size//2 - 1]
    return smoothed

def subtract_moving_average(arr, window_size=11):
    """
    Subtracts the moving average computed over the past 'window_size' time samples.
    """
    filtered = []
    # Start loop at 0 to maintain shape, but average what's available for early frames
    for i in range(arr.shape[1]):
        start_idx = max(0, i - window_size)
        window = arr[:, start_idx:i+1] # Include current frame in avg to avoid offset
        window_mean = np.mean(window, axis=1)
        filtered.append(arr[:, i] - window_mean)
    return np.array(filtered).T

def extract_q_bounds(s_diff, freqs, threshold_db=1.0):
    lower_bounds = []
    upper_bounds = []
    
    for t in range(s_diff.shape[1]):
        frame_data = np.abs(s_diff[:, t])
        peak_idx = np.argmax(frame_data)
        peak_val = frame_data[peak_idx]
        bound_val = peak_val - threshold_db
        
        l_idx = peak_idx
        while l_idx > 0 and frame_data[l_idx] > bound_val:
            l_idx -= 1
            
        u_idx = peak_idx
        while u_idx < len(frame_data) - 1 and frame_data[u_idx] > bound_val:
            u_idx += 1
            
        lower_bounds.append(freqs[l_idx])
        upper_bounds.append(freqs[u_idx])
        
    return np.array(lower_bounds), np.array(upper_bounds)

def plot_tracker_comparison(times, smoothed_res_freqs, smoothed_phase_track, save_path="", keydown_frames=None, keyup_frames=None, calib_end_time=None, title_prefix=""):
    # Apply seaborn theming specifically for this plot
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    
    # Plot using standard plot but taking advantage of seaborn theme
    ax.plot(times, smoothed_res_freqs, color='royalblue', linewidth=3, label='Magnitude Centroid Track')
    ax.plot(times, smoothed_phase_track, color='darkorange', linewidth=3, label='Phase Steepness Track')

    if keydown_frames or keyup_frames:
        keydown_frames = keydown_frames or []
        keyup_frames = keyup_frames or []
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax.axvspan(kd, ku, color='gray', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")
    
    if calib_end_time is not None:
        ax.axvspan(times[0], calib_end_time, facecolor='none', hatch='//', edgecolor='gray', alpha=0.5, label='Calib Window')

    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}Magnitude Change vs. d$\phi$/df Tracking", pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)')
    ax.set_ylabel('Resonant Frequency (MHz)')
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, fontsize='small')
    sns.despine(ax=ax) # Removes top and right borders for scientific look
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)
    
    # Reset seaborn to default matplotlib so it doesn't affect the 4-panel dashboard on subsequent runs (if looping)

def plot_triple_tracker_comparison(times, smoothed_res_freqs, smoothed_phase_track, smoothed_dphi_dt_track, save_path="", keydown_frames=None, keyup_frames=None, calib_end_time=None, title_prefix=""):
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(times, smoothed_res_freqs, color='royalblue', linewidth=3, label='Mag Centroid Track')
    ax.plot(times, smoothed_phase_track, color='darkorange', linewidth=3, label='d$\phi$/df Track')
    ax.plot(times, smoothed_dphi_dt_track, color='mediumseagreen', linewidth=3, label='|d$\phi$/dt| Track')

    if keydown_frames or keyup_frames:
        keydown_frames = keydown_frames or []
        keyup_frames = keyup_frames or []
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax.axvspan(kd, ku, color='gray', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")
    
    if calib_end_time is not None:
        ax.axvspan(times[0], calib_end_time, facecolor='none', hatch='//', edgecolor='gray', alpha=0.5, label='Calib Window')

    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}Resonance Tracking: Mag vs $d\phi/df$ vs $|d\phi/dt|$", pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)')
    ax.set_ylabel('Resonant Frequency (MHz)')
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, fontsize='small')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_3d_waterfall(times, freqs, s_diff_blurred, save_path="", title_prefix=""):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    T, F = np.meshgrid(times, freqs)
    
    surf = ax.plot_surface(T, F, s_diff_blurred, cmap='inferno', edgecolor='none')
    
    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}abs(Differential Modulation Depth) of S21", pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)', labelpad=10)
    ax.set_ylabel('Frequency (MHz)', labelpad=10)
    ax.set_zlabel('Modulation Depth (|dB|)', labelpad=10)
    ax.view_init(elev=25, azim=-60)
    
    fig.colorbar(surf, ax=ax, label='Modulation Depth (|dB|)', shrink=0.5, aspect=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_plotly_waterfall(times, freqs, s_diff_blurred, save_path="", title_prefix=""):
    fig = go.Figure(data=[go.Surface(z=s_diff_blurred, x=times, y=freqs, colorscale='Inferno')])
    fig.update_layout(
        title=dict(text=f"{title_prefix + ' ' if title_prefix else ''}abs(Differential Modulation Depth) of S21", x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title='Time (Milliseconds)',
            yaxis_title='Frequency (MHz)',
            zaxis_title='Modulation Depth (|dB|)'
        ),
        width=1000, height=800)
    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive HTML: {save_path}")

def plot_smith_chart(s_port_raw, freqs, port_label='S21', save_path="", title_prefix=""):
    """
    Plots the S-parameter trajectory on a Smith chart for every frame in the recording.
    Each frame's frequency sweep is drawn as a line colored by time (frame index).
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Draw the Smith chart grid manually using the unit circle + constant-R/X circles
    theta = np.linspace(0, 2 * np.pi, 300)
    # Outer boundary (unit circle)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.2)
    
    # Constant resistance circles: centered at (r/(r+1), 0) radius 1/(r+1)
    for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx = r / (r + 1)
        radius = 1 / (r + 1)
        ax.plot(cx + radius * np.cos(theta), radius * np.sin(theta),
                color='gray', linewidth=0.6, linestyle='--', alpha=0.7)
        ax.text(cx + radius + 0.01, 0.01, f'{r}', fontsize=7, color='gray')
    
    # Constant reactance arcs: these are circles centered at (1, 1/x) with radius 1/x
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            cx, cy = 1, sign / x
            radius = 1 / x
            # Only draw the arc inside the unit circle
            t = np.linspace(0, 2 * np.pi, 500)
            xs = cx + radius * np.cos(t)
            ys = cy + radius * np.sin(t)
            mask = xs**2 + ys**2 <= 1.01
            ax.plot(xs[mask], ys[mask], color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    
    # Real axis
    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.7)
    
    # Plot trajectories: each frame is one S21 frequency sweep
    # s_port_raw shape: (n_freqs, n_frames) complex
    n_frames = s_port_raw.shape[1]
    # Downsample frames to avoid clutter (max 30 lines)
    step = max(1, n_frames // 30)
    frame_indices = np.arange(0, n_frames, step)
    colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))
    
    for i, fidx in enumerate(frame_indices):
        gamma = s_port_raw[:, fidx]  # Complex reflection coefficient for each freq
        # Normalize to unit circle domain
        ax.plot(gamma.real, gamma.imag, color=colors[i], linewidth=0.8, alpha=0.7)
    
    # Colorbar to map colors to frames
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_frames))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Frame Index (Time)')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}Smith Chart: {port_label} Trajectory Over Time", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Real (Resistance)')
    ax.set_ylabel('Imaginary (Reactance)')
    ax.grid(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_notch_frequency(data, freqs_mhz, save_path="", keydown_frames=None, keyup_frames=None, title_prefix=""):
    """
    Tracks and plots the resonant notch (absolute minimum magnitude) for S11 and S21
    over time. Uses raw (non-calibrated) magnitude so the absolute dip is found.
    JUSTIFICATION WILL NEVER BE APPLIED HERE.
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title_prefix + ' ' if title_prefix else ''}Resonant Notch Frequency Tracking", fontweight='bold', fontsize=18)
    
    for ax, port in zip(axes, ['s11', 's21']):
        s_raw = data[port][:, 0].T          # shape: (n_freqs, n_frames)
        s_raw_even = s_raw[:, ::2]           # subsample every other frame
        s_mag_raw = logmag(s_raw_even)       # convert to dB
        
        # argmin: find frequency bin with absolute lowest magnitude each frame
        notch_idx = np.argmin(s_mag_raw, axis=0)
        notch_freqs = freqs_mhz[notch_idx]
        
        # Light smoothing to reduce single-frame noise spikes
        notch_smoothed = smooth_track(notch_freqs, target_window=7)
        
        if 'ts' in data:
            times = data['ts'][::2] * 1000
        else:
            times = np.arange(s_raw_even.shape[1])
        
        ax.plot(times, notch_freqs, color='lightgray', linewidth=1.2, label='Raw Notch')
        ax.plot(times, notch_smoothed, color=sns.color_palette()[0] if port == 's11' else sns.color_palette()[3],
                linewidth=2.5, label=f'{port.upper()} Notch (Smoothed)')

        if keydown_frames or keyup_frames:
            _keydown_frames = keydown_frames or []
            _keyup_frames = keyup_frames or []
            for i, kd in enumerate(_keydown_frames):
                ku = _keyup_frames[i] if i < len(_keyup_frames) else times[-1]
                ax.axvspan(kd, ku, color='gray', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
                ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
                ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")

        ax.set_ylabel('Notch Frequency (MHz)')
        ax.set_title(port.upper())
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        sns.despine(ax=ax)
    
    axes[-1].set_xlabel('Time (Milliseconds)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_magnitude_heatmap(times, freqs, s_diff_blurred, smoothed_track=None, smoothed_lower=None, smoothed_upper=None, save_path="", keydown_frames=None, keyup_frames=None, calib_end_time=None, target_port='s21', title_prefix=""):
    fig, ax = plt.subplots(figsize=(14, 6))
    cp = ax.contourf(times, freqs, s_diff_blurred, levels=30, cmap='crest_r')
    
    if smoothed_track is not None:
        ax.plot(times, smoothed_track, color='cyan', linewidth=2.5, label='Mag Centroid Track')
    if smoothed_lower is not None:
        ax.plot(times, smoothed_lower, color='khaki', linewidth=1, linestyle='--', label='Lower Bound (-1dB)')
    if smoothed_upper is not None:
        ax.plot(times, smoothed_upper, color='khaki', linewidth=1, linestyle='--', label='Upper Bound (-1dB)')

    if keydown_frames or keyup_frames:
        keydown_frames = keydown_frames or []
        keyup_frames = keyup_frames or []
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax.axvspan(kd, ku, facecolor='lightgray', edgecolor='none', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")

    if calib_end_time is not None:
        ax.axvspan(times[0], calib_end_time, facecolor='none', hatch='//', edgecolor='lightgray', alpha=0.5, label='Calib Window')

    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', fontsize='small', framealpha=1, edgecolor='gray')

    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}Magnitude Differential $|{target_port.upper()}|$ with Bandwidth", pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)')
    ax.set_ylabel('Frequency (MHz)')
    
    fig.colorbar(cp, ax=ax, label='Modulation Depth (|dB|)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_phase_grad_heatmap(times, freqs, phase_grad_blurred, smoothed_track=None, save_path="", keydown_frames=None, keyup_frames=None, calib_end_time=None, title_prefix=""):
    fig, ax = plt.subplots(figsize=(14, 6))
    cp = ax.contourf(times, freqs, phase_grad_blurred, levels=30, cmap='flare_r')
    
    if smoothed_track is not None:
        ax.plot(times, smoothed_track, color='darkorange', linewidth=2.5, label='d(Phase)/df Resonance Track')

    if keydown_frames or keyup_frames:
        keydown_frames = keydown_frames or []
        keyup_frames = keyup_frames or []
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax.axvspan(kd, ku, facecolor='lightgray', edgecolor='none', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")

    if calib_end_time is not None:
        ax.axvspan(times[0], calib_end_time, facecolor='none', hatch='//', edgecolor='lightgray', alpha=0.5, label='Calib Window')

    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', fontsize='small', facecolor='#EEEEEE', framealpha=1, edgecolor='gray')

    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}" + r'Phase Gradient Dynamics $d\phi/df$', pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)')
    ax.set_ylabel('Frequency (MHz)')
    
    fig.colorbar(cp, ax=ax, label='Phase Shift Steepness (deg/bin)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_dphi_dt(times, freqs, dphi_dt_blurred, smoothed_track=None, save_path="", keydown_frames=None, keyup_frames=None, calib_end_time=None, title_prefix=""):
    """
    Visualizes the time derivative of phase (dphi/dt) as a 2D heatmap.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cp = ax.contourf(times, freqs, dphi_dt_blurred, levels=30, cmap='viridis')
    if smoothed_track is not None:
        ax.plot(times, smoothed_track, color='mediumspringgreen', linewidth=2.5, label='Max d$\phi$/dt Track')

    if keydown_frames or keyup_frames:
        keydown_frames = keydown_frames or []
        keyup_frames = keyup_frames or []
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax.axvspan(kd, ku, facecolor='lightgray', edgecolor='none', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")

    if calib_end_time is not None:
        ax.axvspan(times[0], calib_end_time, facecolor='none', hatch='//', edgecolor='lightgray', alpha=0.5, label='Calib Window')

    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', fontsize='small', framealpha=1, edgecolor='gray')

    ax.set_title(f"{title_prefix + ' ' if title_prefix else ''}" + r'Phase Derivative over Time $\left|\frac{d\phi}{dt}\right|$', pad=15, fontweight='bold')
    ax.set_xlabel('Time (Milliseconds)')
    ax.set_ylabel('Frequency (MHz)')
    
    fig.colorbar(cp, ax=ax, label='Phase Rate of Change (|deg/frame|)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_sandbox(data, target_port='s21', save_path='', just=False, viz_keypress=False, title_prefix=""):
    s_port = data[target_port][:, 0].T
    # Also extract the alternate port for Smith chart comparison
    other_port = 's11' if target_port == 's21' else 's21'
    s_port_other = data[other_port][:, 0].T
    
    # Magnitudes
    s_mag = logmag(s_port)
    s_even = s_mag[:, ::2]

    # Phases
    s_phase = np.angle(s_port, deg=True)
    s_phase_even = s_phase[:, ::2]

    # Calibration Logic
    if just:
        # Static Baseline Calibration ("Justification")
        baseline_frames = 11
        if s_even.shape[1] < baseline_frames:
            baseline_frames = s_even.shape[1]
            
        s_cal = np.mean(s_even[:, :baseline_frames], axis=1).reshape(-1, 1)
        s_diff = s_even - s_cal
        
        s_phase_cal = np.mean(s_phase_even[:, :baseline_frames], axis=1).reshape(-1, 1)
        # Using np.angle difference handling wrap around
        s_phase_diff = s_phase_even - s_phase_cal
        s_phase_diff = (s_phase_diff + 180) % 360 - 180
    else:
        # Causal Moving Average Filter
        s_diff = subtract_moving_average(s_even, window_size=11)
        
        # We also need to moving average the phase, accounting for angle wrapping
        s_phase_diff_unwrap = np.unwrap(s_phase_even, axis=1, period=360)
        s_phase_diff = subtract_moving_average(s_phase_diff_unwrap, window_size=11)
        s_phase_diff = (s_phase_diff + 180) % 360 - 180

    start_freq = data['start_freq'][0] / 1e6
    stop_freq = data['stop_freq'][0] / 1e6   
    freqs = np.linspace(start_freq, stop_freq, s_diff.shape[0])
    if 'ts' in data:
        times = data['ts'][::2] * 1000
    else:
        times = np.arange(s_diff.shape[1])
    
    # 1. Heatmap Blob smoothing for magnitude
    s_diff_abs = np.abs(s_diff)
    s_diff_blurred = gaussian_filter(s_diff_abs, sigma=(2, 2))
    res_freq_diff_idx = np.argmax(s_diff_blurred, axis=0)
    res_freqs_diff = freqs[res_freq_diff_idx]

    smoothed_res_freqs = smooth_track(res_freqs_diff, target_window=9)

    # Q factor bounds
    lower_bounds, upper_bounds = extract_q_bounds(s_diff_blurred, freqs, threshold_db=2.0)
    smoothed_lower = smooth_track(lower_bounds, target_window=9)
    smoothed_upper = smooth_track(upper_bounds, target_window=9)

    # 2. Phase Derivative Tracking
    # Resonance correlates to max d(Phase)/df
    phase_grad = np.abs(np.gradient(s_phase_diff, axis=0))
    phase_grad_blurred = gaussian_filter(phase_grad, sigma=(2, 2))
    phase_peak_idx = np.argmax(phase_grad_blurred, axis=0)
    phase_peak_freqs = freqs[phase_peak_idx]
    
    # Calculate d(Phase)/dt for separate visualization
    s_phase_unwrap = np.unwrap(s_phase_even, axis=1, period=360)
    if s_phase_unwrap.shape[1] < 2:
        dphi_dt = np.zeros_like(s_phase_unwrap)
    else:
        dphi_dt = np.abs(np.gradient(s_phase_unwrap, axis=1))
    dphi_dt_blurred = gaussian_filter(dphi_dt, sigma=(2, 2))
    
    dphi_dt_peak_idx = np.argmax(dphi_dt_blurred, axis=0)
    dphi_dt_peak_freqs = freqs[dphi_dt_peak_idx]
    smoothed_dphi_dt_track = smooth_track(dphi_dt_peak_freqs, target_window=9)
    smoothed_phase_track = smooth_track(phase_peak_freqs, target_window=9)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"{title_prefix + ' ' if title_prefix else ''}{target_port.upper()} Resonance Dynamics", fontsize=24, y=0.98)
    
    # GROUND TRUTH ADDITION: extract raw keypress times
    keydown_frames = []
    keyup_frames = []
    if viz_keypress:
        if 'keydowns' in data and 'keyups' in data:
            keydown_frames = [kd * 1000 for kd in data['keydowns']]
            keyup_frames = [ku * 1000 for ku in data['keyups']]
        elif 'keypresses' in data:
            keydown_frames = [kp * 1000 for kp in data['keypresses']]

    # Subplot 1: Magnitude Blob
    ax1 = fig.add_subplot(221)
    cp = ax1.contourf(times, freqs, s_diff_blurred, levels=30, cmap='crest_r')
    ax1.plot(times, smoothed_res_freqs, color='cyan', linewidth=2.5, label='Mag Centroid Track')
    ax1.plot(times, smoothed_lower, color='khaki', linewidth=1, linestyle='--', label='Lower Bound (-1dB)')
    ax1.plot(times, smoothed_upper, color='khaki', linewidth=1, linestyle='--', label='Upper Bound (-1dB)')

    # GROUND TRUTH ADDITION: Plot vertical lines for subplot 1
    if keydown_frames or keyup_frames:
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax1.axvspan(kd, ku, facecolor='lightgray', edgecolor='none', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax1.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax1.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")
    if just:
        calib_idx = min(11, len(times)) - 1
        ax1.axvspan(times[0], times[calib_idx], facecolor='none', hatch='//', edgecolor='lightgray', alpha=0.5, label='Calib Window')
    ax1.set_title(f'1. Magnitude Differential $|{target_port.upper()}|$ with Bandwidth')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.legend(loc='lower right', facecolor='#EEEEEE', framealpha=1, edgecolor='gray')
    fig.colorbar(cp, ax=ax1, label='Modulation Depth (|dB|)')
    
    # Subplot 2: Phase Derivative Blob
    ax2 = fig.add_subplot(222)
    cp2 = ax2.contourf(times, freqs, phase_grad_blurred, levels=30, cmap='flare_r')
    ax2.plot(times, smoothed_phase_track, color='darkorange', linewidth=2.5, label='d(Phase)/df Resonance Track')

    # GROUND TRUTH ADDITION: Plot vertical lines for subplot 2
    if keydown_frames or keyup_frames:
        for i, kd in enumerate(keydown_frames):
            ku = keyup_frames[i] if i < len(keyup_frames) else times[-1]
            ax2.axvspan(kd, ku, facecolor='lightgray', edgecolor='none', alpha=0.3, label='Sensor Pressed' if i == 0 else "")
            ax2.axvline(x=kd, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Press' if i == 0 else "")
            ax2.axvline(x=ku, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Release' if i == 0 else "")
    if just:
        calib_idx = min(11, len(times)) - 1
        ax2.axvspan(times[0], times[calib_idx], facecolor='none', hatch='//', edgecolor='lightgray', alpha=0.5, label='Calib Window')
    ax2.set_title(f'2. Phase Gradient Dynamics $d\phi/df$')
    ax2.set_xlabel('Time (Milliseconds)')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.legend(loc='lower right', facecolor='#EEEEEE', framealpha=1, edgecolor='gray')
    fig.colorbar(cp2, ax=ax2, label='Phase Shift Steepness (deg/bin)')

    # Subplot 3: Zoomed-in Snapshots (Event Tracking)
    ax3 = fig.add_subplot(223)
    # Automatically find the single most intense "press" event
    peak_frame = np.unravel_index(np.argmax(s_diff_blurred), s_diff_blurred.shape)[1]
    
    # Take tight snapshots right around this event
    zoomed_snapshots = [
        max(0, peak_frame - 10), 
        max(0, peak_frame - 5), 
        peak_frame, 
        min(len(times)-1, peak_frame + 5), 
        min(len(times)-1, peak_frame + 10)
    ]
    zoomed_colors = sns.color_palette("flare_r", n_colors=len(zoomed_snapshots))
    
    for i, t in enumerate(zoomed_snapshots):
        ax3.plot(freqs, s_diff_blurred[:, t], color=zoomed_colors[i], linewidth=2.5, label=f'Frame {t}')

    ax3.set_title(f'3. Zoomed-In Evolution (Event at Frame {peak_frame})')
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Modulation Depth (|dB|)')
    ax3.legend()
    sns.despine(ax=ax3)

    # Subplot 4: Broad Snapshots (Entire runtime)
    ax4 = fig.add_subplot(224)
    snapshots = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
    colors = sns.color_palette("flare_r", n_colors=len(snapshots))
    for i, t in enumerate(snapshots):
        ax4.plot(freqs, s_diff_blurred[:, t], color=colors[i], linewidth=2, label=f'Frame {t}')

    ax4.set_title(f'4. Broad Evolution $|{target_port.upper()}|$ (Entire Duration)')
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Modulation Depth (|dB|)')
    ax4.legend()
    sns.despine(ax=ax4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)
    
    # Generate the secondary clean tracker comparison plot
    if save_path:
        calib_end_time = times[min(11, len(times)) - 1] if just else None
        
        mag_hm_save_path = save_path.replace(".png", "_mag_heatmap.png")
        plot_magnitude_heatmap(times, freqs, s_diff_blurred, smoothed_track=smoothed_res_freqs, smoothed_lower=smoothed_lower, smoothed_upper=smoothed_upper, save_path=mag_hm_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, calib_end_time=calib_end_time, target_port=target_port, title_prefix=title_prefix)
        
        phase_grad_save_path = save_path.replace(".png", "_phase_grad_heatmap.png")
        plot_phase_grad_heatmap(times, freqs, phase_grad_blurred, smoothed_track=smoothed_phase_track, save_path=phase_grad_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, calib_end_time=calib_end_time, title_prefix=title_prefix)
        
        comp_save_path = save_path.replace(".png", "_comparison.png")
        plot_tracker_comparison(times, smoothed_res_freqs, smoothed_phase_track, save_path=comp_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, calib_end_time=calib_end_time, title_prefix=title_prefix)
        
        triple_comp_save_path = save_path.replace(".png", "_triple_comparison.png")
        plot_triple_tracker_comparison(times, smoothed_res_freqs, smoothed_phase_track, smoothed_dphi_dt_track, save_path=triple_comp_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, calib_end_time=calib_end_time, title_prefix=title_prefix)
        
        waterfall_save_path = save_path.replace(".png", "_waterfall.png")
        plot_3d_waterfall(times, freqs, s_diff_blurred, save_path=waterfall_save_path, title_prefix=title_prefix)
        
        plotly_save_path = save_path.replace(".png", "_interactive.html")
        plot_plotly_waterfall(times, freqs, s_diff_blurred, save_path=plotly_save_path, title_prefix=title_prefix)
        
        smith_save_path = save_path.replace(".png", f"_smith_{target_port}.png")
        plot_smith_chart(s_port, freqs, port_label=target_port.upper(), save_path=smith_save_path, title_prefix=title_prefix)
        
        smith_other_save_path = save_path.replace(".png", f"_smith_{other_port}.png")
        plot_smith_chart(s_port_other, freqs, port_label=other_port.upper(), save_path=smith_other_save_path, title_prefix=title_prefix)
        
        notch_save_path = save_path.replace(".png", "_notch.png")
        plot_notch_frequency(data, freqs, save_path=notch_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, title_prefix=title_prefix)  # freqs already in MHz
        
        dphi_dt_save_path = save_path.replace(".png", "_dphi_dt.png")
        plot_dphi_dt(times, freqs, dphi_dt_blurred, smoothed_track=smoothed_dphi_dt_track, save_path=dphi_dt_save_path, keydown_frames=keydown_frames, keyup_frames=keyup_frames, calib_end_time=calib_end_time, title_prefix=title_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", help="Name of the npz file to process", required=True)
    parser.add_argument("--just", action="store_true", help="Enable static one-time baseline calibration")
    # GROUND TRUTH ADDITION: cli flag for keypress visualization
    parser.add_argument("--viz_keypress", action="store_true", help="Visualize ground truth keypresses")
    parser.add_argument("--prefix", default="", help="String to prepend to all plot titles (e.g. '1mL/min')")
    args = parser.parse_args()

    try:
        data = np.load(args.filename)
        print("Data loaded successfully.")
        out_base = os.path.dirname(os.path.abspath(args.filename))
        filename_base = os.path.basename(args.filename).replace('.npz', '')
        suffix = "_just" if args.just else ""
        # GROUND TRUTH ADDITION: update output filename
        if args.viz_keypress:
            suffix += "_withGndTruth"
        
        plot_sandbox(data, target_port='s21', 
                    save_path=os.path.join(out_base, f"{filename_base}_sandbox{suffix}.png"), 
                    just=args.just, viz_keypress=args.viz_keypress, title_prefix=args.prefix)
    except Exception as e:
        import traceback
        traceback.print_exc()
