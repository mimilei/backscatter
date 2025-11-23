import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import logmag, phase, subtract_moving_average
import joblib

X_all = []
y_all = []

num_classes = 5
data_path = 'data'


for i in range(1, num_classes+1):
    filename = f'{data_path}/data_{i}.npz'
    data = np.load(filename)

    # Load the data; assuming data['s11'] and data['s21'] are arrays where [:,0] gives the desired slice.
    s11 = data['s11'][:, 0].T
    s21 = data['s21'][:, 0].T

    # Compute magnitude and phase (if needed)
    s11_mag = logmag(s11)
    s11_phase = phase(s11)
    s21_mag = logmag(s21)
    s21_phase = phase(s21)

    # Subtract the moving average
    s11_mag = subtract_moving_average(s11_mag, window_size=11)
    s21_mag = subtract_moving_average(s21_mag, window_size=11)

    s11_mag = s11_mag.T
    s21_mag = s21_mag.T

    # Combine features: each sample is [s11_mag, s21_mag]
    X = np.column_stack((s11_mag, s21_mag))
    
    # Create a label array for this file (each sample gets label i)
    y = np.full(s11_mag.shape[0], i)
    
    # Print shapes for debugging (remove later)
    print(f"File {i}: X shape: {X.shape}, y shape: {y.shape}")
    
    X_all.append(X)
    y_all.append(y)

# Stack features and labels from all files
X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

print(f"Total X_all shape: {X_all.shape}")
print(f"Total y_all shape: {y_all.shape}")

# Split data into training and evaluation sets (e.g., 80% train, 20% eval)
X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"Training set: {X_train.shape}, Evaluation set: {X_eval.shape}")


#### SPLIT

# Create and train the ExtraTreesClassifier on the training data
clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("ExtraTreesClassifier has been trained on the training set.")

# Save the trained model to a file
joblib.dump(clf, "models/model.pkl")
print("Model saved to 'models/model.pkl'.")

# Evaluate on the evaluation set
y_pred_eval = clf.predict(X_eval)
y_pred_eval = np.ravel(y_pred_eval)  # Ensure predictions are 1D

# Compute the confusion matrix for the evaluation set
cm = confusion_matrix(y_eval, y_pred_eval)

#### ALL
# # Create and train the ExtraTreesClassifier
# clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
# clf.fit(X_all, y_all)
# print("ExtraTreesClassifier has been trained.")

# # Save the trained model to a file
# joblib.dump(clf, "models/model.pkl")
# print("Model saved to 'models/model.pkl'.")

# # Predict on the same dataset (or on a test set)
# y_pred = clf.predict(X_all)
# y_pred = np.ravel(y_pred)  # Ensure predictions are 1D

# # Compute the confusion matrix
# cm = confusion_matrix(y_all, y_pred)

# Convert counts to percentages per true label (row normalization)
cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

print("Confusion Matrix (Percentage):")
print(cm_percent)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Hand","None","Long tag","Middle tag", "Short tag"],
            yticklabels=["Hand","None","Long tag","Middle tag", "Short tag"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Percentage)')
plt.show()