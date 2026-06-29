import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from .mat file
def load_data_from_mat(mat_filename):
    data = scipy.io.loadmat(mat_filename)
    emg = data['emg']  # EMG signal from electrodes
    stimulus = data['stimulus'].flatten()  # Flatten stimulus into a 1D array
    print(f"EMG shape: {emg.shape}")
    print(f"Stimulus shape: {stimulus.shape}")
    return emg, stimulus

# High-pass filter function
def highpass_filter(emg, cutoff=10, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    emg_filtered = filtfilt(b, a, emg, axis=0)
    return emg_filtered

# Extract trials based on stimulus changes
def extract_trials(emg, stimulus):
    trials = []
    labels = []
    current_label = stimulus[0]
    start_idx = 0
    for i in range(1, len(stimulus)):
        if stimulus[i] != current_label:
            if current_label in [0, 1, 2]:  # Ensure only valid stimulus values are processed
                trial_data = emg[start_idx:i, :]
                trials.append(trial_data)
                labels.append(current_label)
            current_label = stimulus[i]
            start_idx = i
    if current_label in [0, 1, 2]:  # Handle last trial
        trial_data = emg[start_idx:, :]
        trials.append(trial_data)
        labels.append(current_label)
    print(f"Number of trials: {len(trials)}")
    return trials, labels

# Ensure trials are the same length (trim to shortest trial)
def ensure_same_length(trials):
    min_length = min(trial.shape[0] for trial in trials)
    if min_length == 0:
        raise ValueError("At least one trial has zero length.")
    trials_trimmed = [trial[:min_length, :] for trial in trials]
    return trials_trimmed

# Compute frequency domain features (mean and median frequency)
def compute_frequency_domain_features(trial_ch, fs=100):
    N = len(trial_ch)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(trial_ch)
    psd = (np.abs(fft_vals)**2) / N
    if np.sum(psd) == 0:
        return np.array([0, 0])
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    cumulative_psd = np.cumsum(psd)
    total_power = cumulative_psd[-1]
    idx = np.where(cumulative_psd >= total_power/2)[0][0]
    median_freq = freqs[idx]
    return np.array([mean_freq, median_freq])

# Leave-One-Out Cross-Validation using KNN classifier
def leave_one_trial_out_classification(features, labels, K):
    num_trials = len(features)
    correct = 0
    for i in range(num_trials):
        train_indices = [j for j in range(num_trials) if j != i]
        X_train = np.array([features[j] for j in train_indices])
        y_train = np.array([labels[j] for j in train_indices])
        X_test = np.array(features[i]).reshape(1, -1)
        y_test = labels[i]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        if y_pred[0] == y_test:
            correct += 1
    accuracy = correct / num_trials
    return accuracy

# Method 1: Time-domain features using raw EMG data
def run_method_1_time_domain_raw(trials, labels):
    num_channels = trials[0].shape[1]
    K_values = range(1, 20)
    best_acc = 0
    best_ch_accuracies = []

    for ch in range(num_channels):
        features = [trial[:, ch].flatten() for trial in trials]
        accuracies = [leave_one_trial_out_classification(features, labels, K) for K in K_values]
        best_ch_accuracies.append(accuracies)
    return best_ch_accuracies

# Method 2: Frequency-domain features for each electrode
def run_method_2_frequency_domain(trials, labels):
    num_channels = trials[0].shape[1]
    K_values = range(1, 20)
    best_ch_accuracies = []

    for ch in range(num_channels):
        features = [compute_frequency_domain_features(trial[:, ch]) for trial in trials]
        accuracies = [leave_one_trial_out_classification(features, labels, K) for K in K_values]
        best_ch_accuracies.append(accuracies)
    return best_ch_accuracies

# Method 3: Combined features (time-domain and frequency-domain)
def run_method_3_combined(trials, labels):
    num_channels = trials[0].shape[1]
    K_values = range(1, 20)

    # Combined Time-domain features
    time_features = []
    for trial in trials:
        feat_all = [trial[:, ch].flatten() for ch in range(num_channels)]
        time_features.append(np.concatenate(feat_all))

    # Combined Frequency-domain features
    freq_features = []
    for trial in trials:
        feat_all = [compute_frequency_domain_features(trial[:, ch]) for ch in range(num_channels)]
        freq_features.append(np.concatenate(feat_all))

    # Classification for Combined Time-domain
    time_accuracies = [leave_one_trial_out_classification(time_features, labels, K) for K in K_values]

    # Classification for Combined Frequency-domain
    freq_accuracies = [leave_one_trial_out_classification(freq_features, labels, K) for K in K_values]

    return time_accuracies, freq_accuracies

# Main execution block
if __name__ == "__main__":
    try:
        # Load and filter data
        emg, stimulus = load_data_from_mat('subject2.mat')
        emg_filtered = highpass_filter(emg, cutoff=10, fs=100, order=4)

        # Extract trials
        trials, labels = extract_trials(emg_filtered, stimulus)
        trials = ensure_same_length(trials)

        # Method 1: Time-domain using raw EMG data
        best_ch_td_accuracies = run_method_1_time_domain_raw(trials, labels)

        # Method 2: Frequency-domain using frequency features for each electrode
        best_ch_fd_accuracies = run_method_2_frequency_domain(trials, labels)

        # Method 3: Combine all electrodes (both time and frequency features)
        time_accuracies, freq_accuracies = run_method_3_combined(trials, labels)

        # Plot for Method 1 (Time-Domain Raw Data)
        for ch in range(10):
            plt.figure()
            plt.plot(range(1, 20), best_ch_td_accuracies[ch], marker='o')
            plt.title(f"Accuracy vs K for Time-Domain (Channel {ch + 1})")
            plt.xlabel('K')
            plt.ylabel('Accuracy')
            plt.show()

        # Plot for Method 2 (Frequency-Domain)
        for ch in range(10):
            plt.figure()
            plt.plot(range(1, 20), best_ch_fd_accuracies[ch], marker='o')
            plt.title(f"Accuracy vs K for Frequency-Domain (Channel {ch + 1})")
            plt.xlabel('K')
            plt.ylabel('Accuracy')
            plt.show()

        # Plot for Method 3 (Combined Features)
        for ch in range(10):
            plt.figure()
            plt.plot(range(1, 20), time_accuracies, marker='o', label="Time-Domain")
            plt.plot(range(1, 20), freq_accuracies, marker='x', label="Frequency-Domain")
            plt.title(f"Accuracy vs K for Combined Features (Channel {ch + 1})")
            plt.xlabel('K')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
