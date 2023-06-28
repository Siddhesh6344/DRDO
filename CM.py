#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Example code for filtering EEG signals
import scipy.signal as signal

# Define the filter parameters
order = 4  # Filter order
fs = 250  # Sampling frequency
mu_band = [8, 12]  # μ band frequency range
beta_band = [18, 25]  # β band frequency range

# Compute the filter coefficients for the μ band
b_mu, a_mu = signal.butter(order, [mu_band[0] / (fs / 2), mu_band[1] / (fs / 2)], btype='band')
# Apply the filter to the EEG signal
eeg_filtered_mu = signal.lfilter(b_mu, a_mu, eeg_signal)

# Compute the filter coefficients for the β band
b_beta, a_beta = signal.butter(order, [beta_band[0] / (fs / 2), beta_band[1] / (fs / 2)], btype='band')
# Apply the filter to the EEG signal
eeg_filtered_beta = signal.lfilter(b_beta, a_beta, eeg_signal)


# In[ ]:


import pandas as pd
import numpy as np
import mne

# Load the data from the two CSV files
wavelength1_data = pd.read_csv('wavelength1.csv')
wavelength2_data = pd.read_csv('wavelength2.csv')

timestamps = wavelength1_data.iloc[:, 0].values

info_wavelength1 = mne.create_info(ch_names=wavelength1_data.columns[1:].tolist(),
                                   sfreq=sampling_rate, ch_types='fnirs_cw_amplitude')
info_wavelength2 = mne.create_info(ch_names=wavelength2_data.columns[1:].tolist(),
                                   sfreq=sampling_rate, ch_types='fnirs_cw_amplitude')


raw_wavelength1 = mne.io.RawArray(wavelength1_data.iloc[:, 1:].values.T, info_wavelength1)
raw_wavelength2 = mne.io.RawArray(wavelength2_data.iloc[:, 1:].values.T, info_wavelength2)

data_wavelength1 = raw_wavelength1.get_data()[np.newaxis, :, :]
data_wavelength2 = raw_wavelength2.get_data()[np.newaxis, :, :]

# Create a placeholder events array
n_samples = data_wavelength1.shape[2]
event_id = 1  # Use a unique event ID for all events
events = np.array([[0, 0, event_id]])  # Single event at the start of the data

epochs_wavelength1 = mne.EpochsArray(data_wavelength1, info_wavelength1, events=events, tmin=0)
epochs_wavelength2 = mne.EpochsArray(data_wavelength2, info_wavelength2, events=events, tmin=0)

epochs_wavelength1.apply_baseline()
epochs_wavelength2.apply_baseline()



# In[ ]:


def butter_bandpass(lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# In[ ]:


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=-1)
    return y


# In[ ]:


from mne.filter import filter_data

# Define the filter parameters
low_freq = 0.1  # Lower cutoff frequency in Hz
high_freq = 30  # Upper cutoff frequency in Hz
filter_order = 4  # Filter order (4th order in this case)


epochs_wavelength1_filtered = epochs_wavelength1.copy()
epochs_wavelength2_filtered = epochs_wavelength2.copy()

epochs_wavelength1_filtered._data = filter_data(
    epochs_wavelength1_filtered.get_data(), sfreq=epochs_wavelength1.info['sfreq'],
    l_freq=low_freq, h_freq=high_freq, method='iir', iir_params=dict(order=filter_order)
)

epochs_wavelength2_filtered._data = filter_data(
    epochs_wavelength2_filtered.get_data(), sfreq=epochs_wavelength2.info['sfreq'],
    l_freq=low_freq, h_freq=high_freq, method='iir', iir_params=dict(order=filter_order)
)



# In[ ]:



import numpy as np


eeg_normalized_mu = (eeg_filtered_mu - np.mean(eeg_filtered_mu)) / np.std(eeg_filtered_mu)
eeg_normalized_beta = (eeg_filtered_beta - np.mean(eeg_filtered_beta)) / np.std(eeg_filtered_beta)


# In[ ]:




order = 4  
fs = 10  
high_pass_freq = 0.01  
low_pass_freq = 0.2  

# Compute the filter coefficients
b, a = signal.butter(order, [high_pass_freq, low_pass_freq], btype='band', fs=fs)
# Apply the filter to the HbO signal
hbo_filtered = signal.lfilter(b, a, hbo_signal)
# Apply the filter to the HbR signal
hbr_filtered = signal.lfilter(b, a, hbr_signal)


# In[ ]:


# Normalize the HbO signal
hbo_normalized = (hbo_filtered - np.mean(hbo_filtered)) / np.std(hbo_filtered)
# Normalize the HbR signal
hbr_normalized = (hbr_filtered - np.mean(hbr_filtered)) / np.std(hbr_filtered)


# In[ ]:


order = 4  # Filter order
fs = 10  # Sampling frequency
high_pass_freq = 0.01  # High pass filter cutoff frequency
low_pass_freq = 0.2  # Low pass filter cutoff frequency

b, a = signal.butter(order, [high_pass_freq, low_pass_freq], btype='band', fs=fs)
hbo_filtered = signal.lfilter(b, a, hbo_signal)
hbr_filtered = signal.lfilter(b, a, hbr_signal)


# In[ ]:


from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

def apply_csp(X, labels):
    # Compute class-wise covariance matrices
    cov1 = np.cov(X[labels == 0].T)
    cov2 = np.cov(X[labels == 1].T)

    # Regularization parameter
    alpha = 0.1

    # Regularize covariance matrices
    cov1_reg = (1 - alpha) * cov1 + alpha * np.eye(cov1.shape[0])
    cov2_reg = (1 - alpha) * cov2 + alpha * np.eye(cov2.shape[0])

    # Compute eigenvalues and eigenvectors
    _, W = eigh(cov1_reg, cov1_reg + cov2_reg)

    # Normalize spatial filters
    W = StandardScaler().fit_transform(W)

    # Project signals onto spatial filters
    X_csp = np.dot(X, W)

    return X_csp

X_csp = apply_csp(X, labels)


# In[ ]:


import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold


def extract_eeg_features(eeg_data):
    # Implement EEG feature extraction method
    pass

def extract_fnirs_features(fnirs_data):
    # Implement fNIRS feature extraction method
    pass


eeg_features = extract_eeg_features(eeg_data)
fnirs_features = extract_fnirs_features(fnirs_data)



def ssMCCA(X, Y, n_components):
    # Apply ssMCCA using CCA with sparse regularization
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    return X_c, Y_c

# Apply ssMCCA on EEG and fNIRS features
n_components = 2  # Number of components to extract
eeg_c, fnirs_c = ssMCCA(eeg_features, fnirs_features, n_components)

def validate_model(X, Y, model):
    kf = KFold(n_splits=10, shuffle=True)  # Perform 10-fold cross-validation
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        scores.append(score)
    return np.mean(scores)

lasso = Lasso(alpha=0.1) 
score = validate_model(eeg_c, fnirs_c, lasso)
print("Validation score:", score)




# In[ ]:




