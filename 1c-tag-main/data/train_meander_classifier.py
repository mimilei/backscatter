import os
import glob
import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split, LeaveOneGroupOut
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import (logmag, subtract_moving_average, phase_components, rolling_std,
                        fit_noise_reference, mahalanobis_distance,
                        fit_covariance_reference, pairwise_mahalanobis)
import joblib

"""
For Human Skin Test (7/8/2026)
Trains meander liquid classifier. Will either use 80/20 split, 5-fold cross validation,
or leave-one-session-out cross validation depending on flag values.

Features: magnitude + wraparound-safe (cos, sin) phase + phase std for s11/s21, plus a
Mahalanobis-distance-from-noise feature. `noise` is not a classification target -- its
.npz files are loaded separately and used only to fit the Mahalanobis reference.

Examples:
python train_meander_classifier.py --eval_mode split
python train_meander_classifier.py --eval_mode kfold
python train_meander_classifier.py --eval_mode leave_one_session_out
"""

# Configuration
study_name = "20260714"
classes = ['human_presence', 'idle', 'egain']  # noise removed as a classification target
base_data_path = f'data/{study_name}'
model_path = f'models/{study_name}'


def load_file_features(filename):
    """Load one .npz recording and compute its full feature matrix (windowed
    magnitude + phase + phase-std blocks). One row per time-sample frame."""
    data = np.load(filename)
    s11 = data['s11'][:, 0].T
    s21 = data['s21'][:, 0].T

    s11_mag_raw = logmag(s11)
    s21_mag_raw = logmag(s21)
    s11_mag = subtract_moving_average(s11_mag_raw, window_size=11).T
    s21_mag = subtract_moving_average(s21_mag_raw, window_size=11).T

    s11_cos_raw, s11_sin_raw = phase_components(s11)
    s21_cos_raw, s21_sin_raw = phase_components(s21)
    s11_cos = subtract_moving_average(s11_cos_raw, window_size=11).T
    s11_sin = subtract_moving_average(s11_sin_raw, window_size=11).T
    s21_cos = subtract_moving_average(s21_cos_raw, window_size=11).T
    s21_sin = subtract_moving_average(s21_sin_raw, window_size=11).T

    s11_cos_std = rolling_std(s11_cos_raw, window_size=11).T
    s11_sin_std = rolling_std(s11_sin_raw, window_size=11).T
    s21_cos_std = rolling_std(s21_cos_raw, window_size=11).T
    s21_sin_std = rolling_std(s21_sin_raw, window_size=11).T

    return np.column_stack((s11_mag, s21_mag, s11_cos, s11_sin, s21_cos, s21_sin,
                             s11_cos_std, s11_sin_std, s21_cos_std, s21_sin_std))


def print_class_separation_report(X, y, groups, classes, X_noise=None, noise_groups=None):
    """Quantify how different the classes' data really are, and whether that
    separation might just be a same-session artifact rather than a genuine
    physical difference."""
    print("\n=== Class Separation Report ===")

    for label_idx, c in enumerate(classes):
        mask = y == label_idx
        n_rows = mask.sum()
        n_sessions = len(np.unique(groups[mask]))
        print(f"  {c:16s}: {n_rows:5d} rows across {n_sessions} sessions")

    ref = fit_covariance_reference(X)
    precision = ref.precision_

    centroids = {c: X[y == idx].mean(axis=0) for idx, c in enumerate(classes)}

    print("\n  Between-class centroid distances (Mahalanobis):")
    header = "                  " + "".join(f"{c:>16s}" for c in classes)
    print(header)
    for c_row in classes:
        row_str = f"  {c_row:16s}"
        for c_col in classes:
            d = pairwise_mahalanobis(centroids[c_row], centroids[c_col], precision)
            row_str += f"{d:16.2f}"
        print(row_str)

    print("\n  Within-class, across-session spread (session centroids vs. each other):")
    within_spread = {}
    for label_idx, c in enumerate(classes):
        mask = y == label_idx
        session_ids = np.unique(groups[mask])
        session_centroids = [X[mask & (groups == s)].mean(axis=0) for s in session_ids]
        dists = [pairwise_mahalanobis(session_centroids[i], session_centroids[j], precision)
                 for i in range(len(session_centroids)) for j in range(i + 1, len(session_centroids))]
        within_spread[c] = np.mean(dists)
        print(f"    {c:16s}: mean={np.mean(dists):8.2f}  min={np.min(dists):8.2f}  max={np.max(dists):8.2f}")

    print("\n  Separation ratio (between-class distance / max within-class spread of the pair):")
    for i, c_a in enumerate(classes):
        for c_b in classes[i + 1:]:
            between = pairwise_mahalanobis(centroids[c_a], centroids[c_b], precision)
            within = max(within_spread[c_a], within_spread[c_b])
            ratio = between / within
            print(f"    {c_a} vs {c_b}: {ratio:.2f}  (between={between:.2f}, within={within:.2f})")

    if X_noise is not None and noise_groups is not None:
        print("\n  Noise reference QC (within-session spread of the noise baseline itself):")
        session_ids = np.unique(noise_groups)
        session_centroids = [X_noise[noise_groups == s].mean(axis=0) for s in session_ids]
        dists = [pairwise_mahalanobis(session_centroids[i], session_centroids[j], precision)
                 for i in range(len(session_centroids)) for j in range(i + 1, len(session_centroids))]
        print(f"    noise           : mean={np.mean(dists):8.2f}  min={np.min(dists):8.2f}  max={np.max(dists):8.2f}")

    print("=== End Class Separation Report ===\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Real-Time Pressure Classifier")
    parser.add_argument("--eval_mode", choices=["kfold", "split", "leave_one_session_out"], default="kfold",
                         help="Evaluation mode: 'kfold' (5-Fold CV), 'split' (80/20 train/test split), "
                              "or 'leave_one_session_out' (Leave-One-Group-Out CV grouped by recording file)")
    args = parser.parse_args()

    X_all = []
    y_all = []
    groups_all = []

    # Load Data
    print(f"Loading data from {base_data_path}...")
    for label_idx, case in enumerate(classes):
        filenames = sorted(glob.glob(f"{base_data_path}/{case}/data_*.npz"))
        if not filenames:
            print(f"Warning: No files found for {case}, skipping...")
            continue

        for filename in filenames:
            X = load_file_features(filename)
            X_all.append(X)
            y_all.append(np.full(X.shape[0], label_idx))
            groups_all.append(np.full(X.shape[0], filename))

    if len(X_all) == 0:
        print("No data found! Please record data first.")
        exit()

    # Stack features and labels from all files
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups_all = np.concatenate(groups_all)

    # Load noise data separately -- used only to fit the Mahalanobis reference,
    # never as a classification target.
    noise_filenames = sorted(glob.glob(f"{base_data_path}/noise/data_*.npz"))
    if not noise_filenames:
        raise ValueError("No 'noise' data found; cannot compute Mahalanobis reference.")
    noise_file_features = [load_file_features(f) for f in noise_filenames]
    X_noise = np.vstack(noise_file_features)
    noise_groups = np.concatenate([np.full(feat.shape[0], f) for feat, f in zip(noise_file_features, noise_filenames)])

    print_class_separation_report(X_all, y_all, groups_all, classes, X_noise=X_noise, noise_groups=noise_groups)

    noise_reference = fit_noise_reference(X_noise)
    mahal_feature = mahalanobis_distance(X_all, noise_reference).reshape(-1, 1)
    X_all = np.hstack([X_all, mahal_feature])

    print(f"Total dataset shape: X={X_all.shape}, y={y_all.shape}")

    if args.eval_mode == "kfold":
        print("Evaluating model with 5-Fold Stratified Cross-Validation...")
        clf_eval = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_eval = cross_val_predict(clf_eval, X_all, y_all, cv=cv, n_jobs=-1)
        y_true_eval = y_all

        accuracy = np.mean(y_true_eval == y_pred_eval) * 100
        print(f"5-Fold CV Overall Accuracy: {accuracy:.2f}%\n")
    elif args.eval_mode == "split":
        print("Evaluating model with 80/20 Train/Test Split...")
        X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        print(f"Training set: {X_train.shape}, Evaluation set: {X_eval.shape}")
        print("Unique Train Labels:", np.unique(y_train, return_counts=True))
        print("Unique Evaluation Labels:", np.unique(y_eval, return_counts=True))

        clf_eval = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_eval.fit(X_train, y_train)
        y_pred_eval = clf_eval.predict(X_eval)
        y_true_eval = y_eval

        accuracy = np.mean(y_true_eval == y_pred_eval) * 100
        print(f"80/20 Split Overall Accuracy: {accuracy:.2f}%\n")
    elif args.eval_mode == "leave_one_session_out":
        print("Evaluating model with Leave-One-Session-Out Cross-Validation...")
        clf_eval = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv = LeaveOneGroupOut()
        print(f"  {len(np.unique(groups_all))} total recording sessions ({len(np.unique(groups_all))} LOGO folds)")
        y_pred_eval = cross_val_predict(clf_eval, X_all, y_all, cv=cv, groups=groups_all, n_jobs=-1)
        y_true_eval = y_all

        accuracy = np.mean(y_true_eval == y_pred_eval) * 100
        print(f"Leave-One-Session-Out Overall Accuracy: {accuracy:.2f}%\n")

    cm = confusion_matrix(y_true_eval, y_pred_eval)

    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    balanced_acc = balanced_accuracy_score(y_true_eval, y_pred_eval) * 100
    print(f"Balanced accuracy: {balanced_acc:.2f}%")
    print("Per-class recall:")
    for c, r in zip(classes, per_class_recall):
        print(f"  {c:16s}: {r*100:6.2f}%")

    # Train Final Model
    print("Training final model on 100% of the data for deployment...")
    clf_final = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_final.fit(X_all, y_all)

    # Save the trained model
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.pkl")
    joblib.dump({'clf': clf_final, 'noise_reference': noise_reference}, model_file)
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
