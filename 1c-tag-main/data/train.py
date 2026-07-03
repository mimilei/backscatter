import os
import glob
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import logmag, phase, subtract_moving_average, moving_average
import joblib

isClassification = False

num_session = 5
data_path = 'data/study/init_test/'
model_path = data_path.replace('data/', "models/")

# tags = range(1,12+1,2)
distance = ['50cm','100cm','150cm','200cm','250cm']
tags = [0]
# distance = ['all']

if 'cut' in data_path:
    for eval_i in range(1, num_session+1):
        X_all = []
        y_all = []
        for d in distance:
            for i in range(1, num_session+1):
                for t in tags:
                    if eval_i == i:
                        continue
                    filename = f'{data_path}/{d}/{i}/data_{t}.npz'
                    data = np.load(filename)

                    # Load the data; assuming data['s11'] and data['s21'] are arrays where [:,0] gives the desired slice.
                    s11 = data['s11'][:, 0].T
                    s21 = data['s21'][:, 0].T

                    # Compute magnitude and phase (if needed)
                    s11_mag = logmag(s11)
                    s11_phase = phase(s11)
                    s21_mag = logmag(s21)
                    s21_phase = phase(s21)


                    label = [t]*s11_mag.shape[1]
                    cal_data = np.load(f'{data_path}/{d}/{i}/data_0.npz')
                    s11_cal = cal_data['s11'][:, 0].T
                    s21_cal = cal_data['s21'][:, 0].T
                    s11_mag_cal = np.mean(logmag(s11_cal),axis=1).reshape(-1,1)
                    s11_phase_cal = np.mean(phase(s11_cal),axis=1).reshape(-1,1)
                    s21_mag_cal = np.mean(logmag(s21_cal),axis=1).reshape(-1,1)
                    s21_phase_cal = np.mean(phase(s21_cal),axis=1).reshape(-1,1)

                    s11_mag = s11_mag - s11_mag_cal
                    s11_phase = s11_phase - s11_phase_cal
                    s21_mag = s21_mag - s21_mag_cal
                    s21_phase = s21_phase - s21_phase_cal

                    s11_mag = s11_mag.T
                    s21_mag = s21_mag.T

                    # Combine features: each sample is [s11_mag, s21_mag]
                    X = np.column_stack((s11_mag, s21_mag))
                    
                    # Create a label array for this file (each sample gets label i)
                    y = np.array(label)


                    # feature window
                    w = 5
                    X = np.lib.stride_tricks.sliding_window_view(X, (w, X.shape[1]))
                    X = X[:, 0].reshape(X.shape[0], -1)
                    y = y.reshape(-1,1)
                    y = np.lib.stride_tricks.sliding_window_view(y, (w, y.shape[1]))
                    y = y[:, 0].reshape(y.shape[0], -1)[:,0]

                    # Print shapes for debugging (remove later)
                    print(f"File {filename}: X shape: {X.shape}, y shape: {y.shape}")

                    X_all.append(X)
                    y_all.append(y)

        # Stack features and labels from all files
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        print(f"Total X_all shape: {X_all.shape}")
        print(f"Total y_all shape: {y_all.shape}")

        #### Cross Session
        # Create and train the ExtraTreesClassifier
        if isClassification:
            mmodel = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            mmodel.fit(X_all, y_all)
            print("ExtraTreesClassifier has been trained.")
        else:
            mmodel = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1) # , max_depth=5
            mmodel.fit(X_all, y_all)
            print("ExtraTreesRegressor has been trained.")

        # Save the trained model to a file
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(mmodel, f"{model_path}/model_{eval_i}.pkl")
        print(f"Model saved to {model_path}model.pkl.")
else:
    for eval_i in range(1, num_session+1):
        X_all = []
        y_all = []
        for t in tags:
            for d in distance:
                for i in range(1, num_session+1):
                    if eval_i == i:
                        continue
                    filename = f'{data_path}/{t}/{d}/data_{i}.npz'
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

                    try:
                        label = data['label']
                        if len(label) > 0:
                            label, idx_start, idx_end = moving_average(label,ref_len=s11_mag.shape[1],win=11)
                            s11_mag = s11_mag[:, idx_start:]
                            s21_mag = s21_mag[:, idx_start:]
                            # s11_mag = s11_mag[:, idx_start:idx_end]
                            # s21_mag = s21_mag[:, idx_start:idx_end]
                    except:
                        pass

                    s11_mag = s11_mag.T
                    s21_mag = s21_mag.T

                    # Combine features: each sample is [s11_mag, s21_mag]
                    X = np.column_stack((s11_mag, s21_mag))
                    
                    # Create a label array for this file (each sample gets label i)
                    if isClassification:
                        y = np.full(s11_mag.shape[0], t)
                    else:
                        y = np.array(label)
                    

                    # # feature window
                    # w = 5
                    # X = np.lib.stride_tricks.sliding_window_view(X, (w, X.shape[1]))
                    # X = X[:, 0].reshape(X.shape[0], -1)
                    # y = y.reshape(-1,1)
                    # y = np.lib.stride_tricks.sliding_window_view(y, (w, y.shape[1]))
                    # y = y[:, 0].reshape(y.shape[0], -1)[:,0]

                    # # Print shapes for debugging (remove later)
                    # print(f"File {filename}: X shape: {X.shape}, y shape: {y.shape}")

                    X_all.append(X)
                    y_all.append(y)

                    # x, y = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))

                    # # Create two subplots (stacked vertically)
                    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                    # # Plot for s11
                    # pc1 = plt.pcolormesh(x, y, X, shading='nearest', vmin=-1, vmax=1, cmap='seismic')
                    # plt.show()


        # Stack features and labels from all files
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        print(f"Total X_all shape: {X_all.shape}")
        print(f"Total y_all shape: {y_all.shape}")

        # #### In Session

        # # Split data into training and evaluation sets (e.g., 80% train, 20% eval)
        # X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        # print(f"Training set: {X_train.shape}, Evaluation set: {X_eval.shape}")

        # # Create and train the ExtraTreesClassifier on the training data
        # clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
        # clf.fit(X_train, y_train)
        # print("ExtraTreesClassifier has been trained on the training set.")

        # # Save the trained model to a file
        # joblib.dump(clf, "models/model.pkl")
        # print("Model saved to 'models/model.pkl'.")

        # # Evaluate on the evaluation set
        # y_pred_eval = clf.predict(X_eval)
        # y_pred_eval = np.ravel(y_pred_eval)  # Ensure predictions are 1D

        # # Compute the confusion matrix for the evaluation set
        # cm = confusion_matrix(y_eval, y_pred_eval)

        #### Cross Session
        # Create and train the ExtraTreesClassifier
        if isClassification:
            mmodel = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            mmodel.fit(X_all, y_all)
            print("ExtraTreesClassifier has been trained.")
        else:
            mmodel = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1) # , max_depth=5
            mmodel.fit(X_all, y_all)
            print("ExtraTreesRegressor has been trained.")

        # Save the trained model to a file
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(mmodel, f"{model_path}/model_{eval_i}.pkl")
        print(f"Model saved to {model_path}model.pkl.")

# # Predict on the same dataset (or on a test set)
# y_pred = clf.predict(X_all)
# y_pred = np.ravel(y_pred)  # Ensure predictions are 1D

# # Compute the confusion matrix
# cm = confusion_matrix(y_all, y_pred)

# # Convert counts to percentages per true label (row normalization)
# cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

# print("Confusion Matrix (Percentage):")
# print(cm_percent)

# # Plot the confusion matrix using seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
#             xticklabels=["Hand","None","Long tag","Middle tag", "Short tag"],
#             yticklabels=["Hand","None","Long tag","Middle tag", "Short tag"])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix (Percentage)')
# plt.show()