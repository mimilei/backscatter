import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import logmag, phase, subtract_moving_average, moving_average
import joblib
from scipy import stats

isClassification = False

num_session = 5
model_path = 'models/study/continuous/'
data_path = model_path.replace('models/', "data/")

# tags = range(1,8+1)
distance = ['50cm','100cm','150cm','200cm','250cm']
tags = [0]
# distance = ['all']

distance_labels = []
y_pred_all = []
y_gt_all = []
X_all_all = []
acc_all = []

if 'cut' in model_path:
    for eval_i in range(1, num_session+1):
        X_all = []
        y_all = []
        mmodel = joblib.load(f"{model_path}/model_{eval_i}.pkl")
        for d in distance:
            for t in tags:
                filename = f'{data_path}/{d}/{eval_i}/data_{t}.npz'
                data = np.load(filename)

                # Load the data; assuming data['s11'] and data['s21'] are arrays where [:,0] gives the desired slice.
                s11 = data['s11'][:, 0].T
                s21 = data['s21'][:, 0].T

                # Compute magnitude and phase (if needed)
                s11_mag = logmag(s11)
                s11_phase = phase(s11)
                s21_mag = logmag(s21)
                s21_phase = phase(s21)

                # 86.5 mm
                label = [t]*s11_mag.shape[1]
                cal_data = np.load(f'{data_path}/{d}/{eval_i}/data_0.npz')
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
                
                X_all.append(X)
                y_all.append(y)
                for _ in range(len(np.vstack(X))):
                    distance_labels.append(d)

        # Stack features and labels from all files
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        # Print shapes for debugging (remove later)
        print(f"Total evaluation instances {y_all.shape}")

        y_pred = mmodel.predict(X_all)
        y_pred = np.ravel(y_pred)

        y_pred_all.append(y_pred)
        y_gt_all.append(y_all)
        X_all_all.append(X_all)
else:
    for model_i in range(1,num_session+1):
        X_all = []
        y_all = []
        mmodel = joblib.load(f"{model_path}/model_{model_i}.pkl")
        for d in distance:
            for t in tags:
                filename = f'{data_path}/{t}/{d}/data_{model_i}.npz'
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
                
                X_all.append(X)
                y_all.append(y)
                for _ in range(len(np.vstack(X))):
                    distance_labels.append(d)

        # Stack features and labels from all files
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        # Print shapes for debugging (remove later)
        print(f"Total evaluation instances {y_all.shape}")

        y_pred = mmodel.predict(X_all)
        y_pred = np.ravel(y_pred)

        if isClassification:
            acc = np.sum(y_all==y_pred)/len(y_pred)
            acc_all.append(acc)

        y_pred_all.append(y_pred)
        y_gt_all.append(y_all)
        X_all_all.append(X_all)

# Compute the confusion matrix
y_pred_all = np.concatenate(y_pred_all).flatten()
y_gt_all = np.concatenate(y_gt_all).flatten()
X_all_all = np.concatenate(X_all_all)
distance_labels = np.array(distance_labels)

if isClassification:
    cm = confusion_matrix(y_gt_all, y_pred_all)

    # Convert counts to percentages per true label (row normalization)
    cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    # print("Confusion Matrix (Percentage):")
    # print(cm_percent)
    # print("accuracy:",np.sum(y_gt_all==y_pred_all)/len(y_pred_all))
    print(f"accuracy: {np.mean(acc_all)} (SE={stats.sem(acc_all)}, SD={np.std(acc_all)})")

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=tags,
                yticklabels=tags)
    plt.xlabel('Predicted Tag Label')
    plt.ylabel('Groudtruth Tag Label')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()

    #### Compute accuracy in distance
    acc_in_distance = []
    for i, d in enumerate(distance):
        # _X = X_all_all[distance_labels==d]
        _y = y_gt_all[distance_labels==d]
        # y_pred = mmodel.predict(_X)
        y_pred = y_pred_all[distance_labels==d]
        acc = (_y==y_pred)
        acc = np.sum(acc)/len(acc)
        acc_in_distance.append(acc)
        print(d, acc)
    print("accuracy:",acc)

    plt.figure(figsize=(6,4))
    plt.plot(distance, acc_in_distance, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.ylim(0, 1)  # accuracy range
    plt.xlabel("Distance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Distance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

else:
    # X_all_all = X_all_all[:,101:].T
    # x, y = np.meshgrid(np.arange(X_all_all.shape[1]), np.arange(X_all_all.shape[0]))

    # # Create two subplots (stacked vertically)
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # # Plot for s11
    # pc1 = plt.pcolormesh(x, y, X_all_all, shading='nearest', vmin=-1, vmax=1, cmap='seismic')
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    if 'tracking' in model_path:
        y_pred_all = -y_pred_all * (250 - 50) / 0.85
        y_gt_all = -y_gt_all * (250 - 50) / 0.85
        fig, ax = plt.subplots()
        ax.set_xlim(20, 250)
        ax.set_ylim(20, 250)

        # Scatter plot
        print(y_gt_all.shape, y_pred_all.shape)
        ax.scatter(y_gt_all, y_pred_all, c="#00A1FF", s=50, label="Data", marker="x", alpha=0.3)

        # Perfect linear line (y = x)
        ax.plot([20, 250], [20, 250], 'k--', label="y = x", c="#1C64E7")

        # Fit regression line
        model = LinearRegression()
        condition = y_gt_all<140
        model.fit(y_gt_all[condition].reshape(-1, 1), y_pred_all[condition])
        y_fit = model.predict(y_gt_all[condition].reshape(-1, 1))

        # Plot fitted line
        ax.plot(y_gt_all[condition], y_fit, 'r-', linewidth=2, label="Fitted line", c="#FF42A1")

        # Compute R^2
        r2 = r2_score(y_pred_all[condition], y_fit)
        # ax.text(40, 200, f"$R^2$ = {r2:.2f}", fontsize=12, color="red")

        # Labels and legend
        ax.set_xlabel('Ground truth (cm)')
        ax.set_ylabel('Prediction (cm)')
        # ax.set_aspect('equal')
        ax.legend()


        err = np.abs(y_gt_all[condition]-y_pred_all[condition])
        print(f"error: {np.mean(err)} (SE={stats.sem(err)}), SD={np.std(err)})")

        plt.show()
    elif 'cut' in model_path:
        err_in_distance = []
        std_in_distance = []
        for i, d in enumerate(distance):
            _y = y_gt_all[distance_labels==d]
            y_pred = y_pred_all[distance_labels==d]
            err = np.abs(y_pred - _y)*0.6
            err_in_distance.append(np.mean(err))
            std_in_distance.append(stats.sem(err))
            print(f"{d}, error: {np.mean(err)} (SE={stats.sem(err)}, SD={np.std(err)}))")

        print(f"error: {np.mean(err_in_distance)} (SD={np.mean(std_in_distance)}))")
        
        fig, ax = plt.subplots()
        bars = ax.bar(distance, err_in_distance, yerr=std_in_distance, capsize=5, alpha=0.7, color='skyblue')
        for bar, value in zip(bars, err_in_distance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2-0.2, height + 0.05,  # position
                    f'{value:.1f}', ha='center', va='bottom', fontsize=15)

        x = np.arange(len(distance))
        ax.set_xticks(x)
        ax.set_xticklabels(distance)
        ax.set_ylim(0,1.2)
        plt.xlabel('Antenna to Tag Distance')
        plt.ylabel('Length Error (cm)')
        plt.show()

    elif 'continuous' in model_path:
        err_in_distance = []
        std_in_distance = []
        y_gt_all = (y_gt_all+0.12)/(-0.19+0.12) * 5.6
        y_pred_all = (y_pred_all+0.12)/(-0.19+0.12) * 5.6

        for i, d in enumerate(distance):
            _y = y_gt_all[distance_labels==d]
            y_pred = y_pred_all[distance_labels==d]
            err = np.abs(y_pred - _y)
            err_in_distance.append(np.mean(err))
            std_in_distance.append(np.std(err))

            print("d",np.mean(err), "SE",stats.sem(err), "SD",np.std(err))

            fig, ax = plt.subplots()
            ax.set_xlim(0, 7)
            ax.set_ylim(0, 7)
            # Scatter plot
            ax.scatter(_y, y_pred, s=1, label="Data")
            # Labels and legend
            ax.set_xlabel('Ground truth (cm)')
            ax.set_ylabel('Prediction (cm)')
            ax.set_aspect('equal')
            ax.legend()
            plt.show()

        print(f"error: {np.mean(err_in_distance)} (SD={np.mean(std_in_distance)}))")
        
        fig, ax = plt.subplots()
        bars = ax.bar(distance, err_in_distance, yerr=std_in_distance, capsize=5, alpha=0.7, color='skyblue')
        for bar, value in zip(bars, err_in_distance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2-0.2, height + 0.05,  # position
                    f'{value:.1f}', ha='center', va='bottom', fontsize=15)

        x = np.arange(len(distance))
        ax.set_xticks(x)
        ax.set_xticklabels(distance)
        # ax.set_ylim(0,3)
        plt.xlabel('Antenna to Tag Distance')
        plt.ylabel('Length Error (cm)')
        plt.show()

