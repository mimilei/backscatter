import os
import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import logmag, subtract_moving_average
import joblib

"""
Trains sensor classifier. Will either use 80/20 split or 5-fold cross validation
depending on flag values. 

Examples:
python train_pressure_classifier.py --eval_mode split
python train_pressure_classifier.py --eval_mode kfold
"""

# Configuration
study_name = "20260403_test"
num_session = 5
classes = ['noise', 'human_presence', 'pressure']
base_data_path = f'data/{study_name}'
model_path = f'models/{study_name}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Real-Time Pressure Classifier")
    parser.add_argument("--eval_mode", choices=["kfold", "split"], default="kfold", help="Evaluation mode: 'kfold' (5-Fold CV) or 'split' (80/20 train/test split)")
    args = parser.parse_args()

    X_all = []
    y_all = []

    # Load Data
    print(f"Loading data from {base_data_path}...")
    for label_idx, case in enumerate(classes):
        for i in range(1, num_session + 1):
            # Filename based on feedback: data/{study_name}/{case_name}/{session_number}.npz
            filename = f"{base_data_path}/{case}/data_{i}.npz"
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping...")
                continue
                
            data = np.load(filename)

            # Load the data; assuming data['s11'] and data['s21'] are arrays where [:,0] gives the desired slice.
            s11 = data['s11'][:, 0].T
            s21 = data['s21'][:, 0].T

            # Compute magnitude
            s11_mag = logmag(s11)
            s21_mag = logmag(s21)

            # Subtract the moving average (calibration/baseline removal)
            s11_mag = subtract_moving_average(s11_mag, window_size=11)
            s21_mag = subtract_moving_average(s21_mag, window_size=11)

            s11_mag = s11_mag.T
            s21_mag = s21_mag.T

            # Combine features: each sample is [s11_mag, s21_mag]
            X = np.column_stack((s11_mag, s21_mag))
            
            # Labels are simply the index of the class
            y = np.full(s11_mag.shape[0], label_idx)

            X_all.append(X)
            y_all.append(y)

    if len(X_all) == 0:
        print("No data found! Please record data first.")
        exit()

    # Stack features and labels from all files
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"Total dataset shape: X={X_all.shape}, y={y_all.shape}")

    if args.eval_mode == "kfold":
        print("Evaluating model with 5-Fold Stratified Cross-Validation...")
        clf_eval = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_eval = cross_val_predict(clf_eval, X_all, y_all, cv=cv, n_jobs=-1)
        y_true_eval = y_all
        
        accuracy = np.mean(y_true_eval == y_pred_eval) * 100
        print(f"5-Fold CV Overall Accuracy: {accuracy:.2f}%\n")
    else:
        print("Evaluating model with 80/20 Train/Test Split...")
        X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        print(f"Training set: {X_train.shape}, Evaluation set: {X_eval.shape}")
        
        clf_eval = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_eval.fit(X_train, y_train)
        y_pred_eval = clf_eval.predict(X_eval)
        y_true_eval = y_eval
        
        accuracy = np.mean(y_true_eval == y_pred_eval) * 100
        print(f"80/20 Split Overall Accuracy: {accuracy:.2f}%\n")
        
    cm = confusion_matrix(y_true_eval, y_pred_eval)

    # Train Final Model
    print("Training final model on 100% of the data for deployment...")
    clf_final = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_final.fit(X_all, y_all)

    # Save the trained model
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.pkl")
    joblib.dump(clf_final, model_file)
    print(f"Final model saved to {model_file}")

    # Confusion Matrix visualization
    # Convert counts to percentages per true label (row normalization)
    cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    print("Confusion Matrix (Percentage):")
    print(cm_percent)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({study_name})')
    plt.tight_layout()
    plt.show()
