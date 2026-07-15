import time
import numpy as np
import joblib
from collections import deque
import sys
import warnings

# Benign: TwoPortOnePath calibration in demo.py is only given a single
# measurement orientation, so skrf warns on every apply_cal() call. We only
# consume the fully-corrected S11/S21 (forward) terms, so this is expected.
warnings.filterwarnings(
    'ignore',
    message='only gave a single measurement orientation.*',
    category=UserWarning,
    module='skrf.calibration.calibration',
)

"""
Real-time evaluation script for meander on human skin classifier.

This script connects to a VNA, reads S11 and S21 data, 
applies the same moving average
subtraction as the training script, and uses the trained classifier 
to predict the current state (meander activated, human presence, nacl).

Raw predictions are added to a deque of size 5 (empirically determined), 
and the mode of the deque is printed as the actual classification decision.

Features: magnitude + wraparound-safe (cos, sin) phase + phase std for s11/s21, plus a
Mahalanobis-distance-from-noise feature (loaded from the trained model bundle). `noise`
is not a predicted class -- it's used only to fit the Mahalanobis reference.

Examples:
python eval_meanderOnSkin.py
"""

# Import VNA connectivity from the existing demo.py in the same directory
try:
    # from demo import get_vna_devices, dotdict
    from record_data import get_vna_devices, dotdict
except ImportError:
    print("Error: Could not import utils from record_data. Please ensure you run this from the top-level directory.")
    sys.exit(1)

from read_data import logmag, mahalanobis_distance, phase_components

def main(opt):
    # Match the configuration exactly from your train_pressure_classifier.py
    study_name = "20260714"
    classes = ['human_presence', 'idle', 'egain']
    model_path = f"models/{study_name}/model.pkl"

    print(f"Loading model from {model_path}...")
    try:
        bundle = joblib.load(model_path)
        clf, noise_reference = bundle['clf'], bundle['noise_reference']
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}.")
        print("Please train it first.")
        sys.exit(1)

    print("Connecting to VNA...")
    devices = get_vna_devices(opt)
    if not devices:
        print("No VNA devices found!")
        sys.exit(1)
        
    nv = devices[0] # Using the first connected VNA device
    print(f"Successfully connected to VNA.")
    
    # We use a moving average window of 11 to exactly emulate `subtract_moving_average` 
    # applied during offline training.
    window_size = 11
    buffer_s11 = deque(maxlen=window_size)
    buffer_s21 = deque(maxlen=window_size)
    buffer_s11_cos = deque(maxlen=window_size)
    buffer_s11_sin = deque(maxlen=window_size)
    buffer_s21_cos = deque(maxlen=window_size)
    buffer_s21_sin = deque(maxlen=window_size)

    recent_predictions = deque(maxlen=8)
    
    print("\n--- Starting real-time classification ---")
    print("Waiting to fill moving-average buffer (11 frames)...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Fetch current sweep reading (S11, S21)
            s11, s21 = nv.get_data_()
            
            # Convert to log-magnitude and wraparound-safe phase components
            s11_mag = logmag(s11)
            s21_mag = logmag(s21)
            s11_cos, s11_sin = phase_components(s11)
            s21_cos, s21_sin = phase_components(s21)

            # Form moving average subtraction BEFORE appending the new reading
            # This matches how the causal moving average filter works in the training script.
            if len(buffer_s11) == window_size:
                feat_s11 = s11_mag - np.mean(buffer_s11, axis=0)
                feat_s21 = s21_mag - np.mean(buffer_s21, axis=0)
                feat_s11_cos = s11_cos - np.mean(buffer_s11_cos, axis=0)
                feat_s11_sin = s11_sin - np.mean(buffer_s11_sin, axis=0)
                feat_s21_cos = s21_cos - np.mean(buffer_s21_cos, axis=0)
                feat_s21_sin = s21_sin - np.mean(buffer_s21_sin, axis=0)

                feat_s11_cos_std = np.std(buffer_s11_cos, axis=0)
                feat_s11_sin_std = np.std(buffer_s11_sin, axis=0)
                feat_s21_cos_std = np.std(buffer_s21_cos, axis=0)
                feat_s21_sin_std = np.std(buffer_s21_sin, axis=0)

                raw_feat = np.concatenate([feat_s11, feat_s21, feat_s11_cos, feat_s11_sin, feat_s21_cos, feat_s21_sin,
                                            feat_s11_cos_std, feat_s11_sin_std, feat_s21_cos_std, feat_s21_sin_std])

                # Model expects shape (n_samples, n_features)
                mahal_feat = mahalanobis_distance(raw_feat.reshape(1, -1), noise_reference)
                X = np.concatenate([raw_feat, mahal_feat]).reshape(1, -1)
                
                pred_idx = clf.predict(X)[0]
                prediction = classes[pred_idx]
                
                recent_predictions.append(prediction)
                
                # Get the mode of the last 5 predictions
                preds_list = list(recent_predictions)
                mode_prediction = max(preds_list, key=preds_list.count)
                
                # Output prediction, pad with whitespace to overwrite older text if shorter
                print(f"Current Status: [ {mode_prediction.upper()} ]                ", end='\r')
                
            buffer_s11.append(s11_mag)
            buffer_s21.append(s21_mag)
            buffer_s11_cos.append(s11_cos)
            buffer_s11_sin.append(s11_sin)
            buffer_s21_cos.append(s21_cos)
            buffer_s21_sin.append(s21_sin)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog: [options]")
    # Base configuration arguments required by demo.py's VNAObject setup
    # Defaults set to match the pressure classifier training environment (500MHz to 2.5GHz)
    parser.add_option("-S", "--start", dest="start", type="float", default=0.5e9, help="start frequency")
    parser.add_option("-E", "--stop", dest="stop", type="float", default=2.5e9, help="stop frequency")
    parser.add_option("-N", "--points", dest="points", type="int", default=101, help="scan points")
    (opt, args) = parser.parse_args()
    
    main(opt)
