import time
import numpy as np
import joblib
from collections import deque
import sys

"""
Real-time evaluation script for pressure sensor classifier.

This script connects to a VNA, reads S11 and S21 data, 
applies the same moving average
subtraction as the training script, and uses the trained classifier 
to predict the current state (noise, human_presence, or pressure).

Raw predictions are added to a deque of size 5 (empirically determined), 
and the mode of the deque is printed as the actual classification decision.

Examples:
python eval_pressure.py
"""

# Import VNA connectivity from the existing demo.py in the same directory
try:
    from demo import get_vna_devices, dotdict
except ImportError:
    print("Error: Could not import from demo.py. Please ensure you run this from the top-level directory.")
    sys.exit(1)

def logmag(x):
    return 20 * np.log10(np.abs(x))

def main(opt):
    # Match the configuration exactly from your train_pressure_classifier.py
    study_name = "20260403_test"
    classes = ['noise', 'human_presence', 'pressure']
    model_path = f"models/{study_name}/model.pkl"
    
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}.")
        print("Please train it first using python data/train_pressure_classifier.py")
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
    
    recent_predictions = deque(maxlen=5)
    
    print("\n--- Starting real-time classification ---")
    print("Waiting to fill moving-average buffer (11 frames)...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Fetch current sweep reading (S11, S21)
            s11, s21 = nv.get_data_()
            
            # Convert to log-magnitude
            s11_mag = logmag(s11)
            s21_mag = logmag(s21)
            
            # Form moving average subtraction BEFORE appending the new reading
            # This perfectly matches how the causal moving average filter works in the training script.
            if len(buffer_s11) == window_size:
                s11_mean = np.mean(buffer_s11, axis=0)
                s21_mean = np.mean(buffer_s21, axis=0)
                
                feat_s11 = s11_mag - s11_mean
                feat_s21 = s21_mag - s21_mean
                
                # Model expects shape (n_samples, n_features)
                X = np.concatenate([feat_s11, feat_s21]).reshape(1, -1)
                
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
