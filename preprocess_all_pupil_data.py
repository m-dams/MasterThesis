import os

from pymef import MefSession
from bids import BIDSLayout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, variation
import mne

from scipy.signal import butter, filtfilt

DATA_FOR_LSTM = True
APPLY_LOW_PASS_FILTER = True
FEATURE_SET = 1


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def split_list(lst, n):
    """Split the list, lst, into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def check_missing_values(arr):
    """Check if a numpy array has missing values."""
    if np.isnan(arr).any():
        print("The array has missing values.")
        return True
    else:
        return False


how_many_nan = 0
subject_run_tuples = [('003', 1), ('003', 2), ('004', 1), ('004', 2), ('005', 1), ('005', 2), ('005', 3), ('005', 4),
                      ('006', 1), ('007', 1), ('009', 1), ('009', 2), ('011', 1), ('011', 2), ('012', 1), ('012', 2),
                      ('012', 3), ('012', 4)]
for srt in subject_run_tuples:
    APPLY_LOW_PASS_FILTER = True
    LSTM_READY = False
    FEATURE_SET = 3
    subject = srt[0]
    run = srt[1]

    path_to_dataset = r"C:\MasterThesis\v1.0"  # Please change this value

    l = BIDSLayout(path_to_dataset)

    filter_dictionary = {'subject': subject,
                         'session': '001',
                         'task': 'FR',
                         'run': run,
                         'suffix': 'ieeg',
                         'extension': 'json'}
    json_files = l.get(**filter_dictionary)

    # We now we have requested only one file now but multiple files can be processed in a for loop
    json_file = json_files[0]
    print(json_file)

    json_entities = json_file.entities

    channel_file = l.get(suffix='channels', extension='tsv',
                         task=json_entities['task'],
                         subject=json_entities['subject'],
                         session=json_entities['session'],
                         run=json_entities['run'])[0]
    channel_df = channel_file.get_df()
    channel_df = channel_df[channel_df['status'] != 'bad']
    channel_df

    events_file = l.get(suffix='events', extension='tsv',
                        task=json_entities['task'],
                        subject=json_entities['subject'],
                        session=json_entities['session'],
                        run=json_entities['run'])[0]
    events_df = events_file.get_df()

    # Pull only the processed events (discarding the original events)
    events_df = events_df.loc[~events_df['trial_type'].isna(), ['onset', 'sample', 'duration', 'trial_type', 'list']]
    events_df = events_df[events_df['trial_type'] == 'ENCODE']
    events_df = events_df.reset_index(drop=True)

    # Since we are using MEF3 pybids has problems recognizing the extension so we need to modify the json extension
    mef_session_path = os.path.splitext(json_file.path)[0] + '.mefd'
    print(mef_session_path)
    ms = MefSession(mef_session_path, None)
    ms.read_ts_channel_basic_info()

    channels = ['LEFT_PUPIL_SIZE', 'RIGHT_PUPIL_SIZE']

    # Microseconds 1 μs = 10⁻⁶ s
    start_time = 1553941347170839 + 3 * 1e6  # in microseconds
    end_time = 1553942904095135 - 3 * 1e6
    win_size = 10  # in seconds
    stop_time = start_time + int(win_size * 1e6)

    bi = ms.read_ts_channel_basic_info()

    # In this case we are reading MEF records instead of data from events.tsv because we do not have to make time conversions this way
    records = ms.read_records()
    records_in_win = [x for x in records if start_time < x['time'] < stop_time]
    # The code to get corresponding events from the events file
    session_metadata = ms.session_md
    session_start_utc = session_metadata['session_specific_metadata']['earliest_start_time'][0]
    print(f"session_start_utc = {session_start_utc}\nsession_stop_utc = {stop_time}")
    events_df['microsecond_onset'] = events_df['onset'] * 1e6
    events_df['uutc'] = (events_df['microsecond_onset'] + session_start_utc).astype('int64')
    events_in_win = events_df[(start_time < events_df['uutc'])
                              & (events_df['uutc'] < stop_time)]

    # records_in_events = [x for x in records if (0 < (x['time'] - events_df['uutc']) < 150000)]
    records_in_events = []
    for x in records:
        for e in events_df['uutc']:
            # print(x['time'] - e)
            if 0 == (x['time'] - e):
                records_in_events.append(x)

    len(records_in_events)
    data_events = []
    # −200ms to 0ms from the onset and from 1000ms to 1400ms
    for e in events_df['uutc']:
        data_events.append(ms.read_ts_channels_uutc(channels, [e - 600000, e + 900000]))
    data_events = np.array(data_events)


    def check_missing_values(arr):
        """Check if a numpy array has missing values."""
        if np.isnan(arr).any():
            print("The array has missing values.")
            return True
        else:
            return False


    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            missing_values_idx = []
            for sample_idx, sample in enumerate(channel):
                if np.isnan(sample):
                    missing_values_idx.append(sample_idx)
            print(missing_values_idx)
            for missing in missing_values_idx[::-1]:
                if missing >= 5:
                    for i in range(5):
                        data_events[s_idx][ch_idx][missing - i] = np.nan
                if missing <= 221:
                    for i in range(3):
                        data_events[s_idx][ch_idx][missing + i] = np.nan
    how_many_nan = 0
    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            if check_missing_values(channel):
                how_many_nan += 1
                df = pd.DataFrame({'signal': channel})
                print(f"Apply linear interpolation to fill the missing values [{s_idx}][{ch_idx}]")
                df['signal'] = df['signal'].bfill().ffill().interpolate(method='linear')
                data_events[s_idx][ch_idx] = df['signal'].values.tolist()
                print(data_events[s_idx][ch_idx])

    channel_1_raw = []
    channel_2_raw = []
    for s_idx, sequence in enumerate(data_events):
        for sample in data_events[s_idx][0]:
            channel_1_raw.append(sample)
        for sample in data_events[s_idx][1]:
            channel_2_raw.append(sample)
    channel_1_raw = np.array(channel_1_raw)
    channel_2_raw = np.array(channel_2_raw)

    mean = np.mean(channel_1_raw)
    std = np.std(channel_1_raw)
    data_normalized = (channel_1_raw - mean) / std
    channel_1_raw = data_normalized

    mean = np.mean(channel_2_raw)
    std = np.std(channel_2_raw)
    data_normalized = (channel_2_raw - mean) / std
    channel_2_raw = data_normalized


    def split_list(lst, n):
        """Split the list, lst, into chunks of size n."""
        return [lst[i:i + n] for i in range(0, len(lst), n)]


    samples_in_window = 225
    chunks_1 = split_list(channel_1_raw, samples_in_window)
    chunks_2 = split_list(channel_2_raw, samples_in_window)
    print(len(chunks_1))
    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            if ch_idx == 0:
                data_events[s_idx][ch_idx] = chunks_1[s_idx]
            elif ch_idx == 1:
                data_events[s_idx][ch_idx] = chunks_2[s_idx]
    data_events = np.array(data_events)
    print(data_events.shape)

    print(data_events[0][0][0])


    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a


    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    order = 3
    fs = 150.0  # sample rate, Hz
    cutoff = 20.0  # desired cutoff frequency of the filter, Hz

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            y = butter_lowpass_filter(channel, cutoff, fs, order)
            data_events[s_idx][ch_idx] = y

    sfreq = 150  # replace with your sampling frequency
    # delta: 2.5-5 Hz, theta: 4–9 Hz, alpha: 9–16 Hz, beta: 16–25 Hz,
    freqs = [9., 16., 25., 40.]  # define the range of frequency bands
    # freqs = [5.]
    print(freqs)
    n_cycles = 7.0

    # perform Morlet Wavelet transform
    power = mne.time_frequency.tfr_array_morlet(data_events, sfreq, freqs, n_cycles=n_cycles)
    power = np.mean(power, axis=1)

    if FEATURE_SET == 2 and not LSTM_READY:
        data_events = data_events.tolist()
        # Assuming 'eeg_data' is a NumPy array containing your EEG data.
        for s_idx, sequence in enumerate(data_events):
            for ch_idx, channel in enumerate(sequence):
                eeg_data = channel
                # Assuming 'eeg_data' is a NumPy array containing your EEG data for one channel

                mean = np.mean(eeg_data)
                q1 = np.percentile(eeg_data, 25)
                q2 = np.percentile(eeg_data, 50)
                q3 = np.percentile(eeg_data, 75)
                min_signal = np.min(eeg_data)
                max_signal = np.max(eeg_data)
                # variance = variation(eeg_data)
                # skewness = skew(eeg_data)

                features = [mean, q1, q2, q3, min_signal, max_signal]

                # print("Extracted features:", features)
                data_events[s_idx][ch_idx] = features
        data_events = np.array(data_events)
        print(data_events.shape)

    if FEATURE_SET == 3:
        data_events = data_events.tolist()
        # Assuming 'eeg_data' is a NumPy array containing your EEG data.
        for s_idx, sequence in enumerate(data_events):
            for ch_idx, channel in enumerate(sequence):
                eeg_data = channel
                # Assuming 'eeg_data' is a NumPy array containing your EEG data for one channel

                mean = np.mean(eeg_data)
                variance = variation(eeg_data)
                skewness = skew(eeg_data)
                # kurtosis = kurtosis(eeg_data)

                features = [mean, variance, skewness]

                print("Extracted features:", features)
                data_events[s_idx][ch_idx] = features
        data_events = np.array(data_events)
        print(data_events.shape)

    power_magnitude = np.abs(power)
    # convert to a decibel scale, which is often done for power spectral densities:
    power_db = 10 * np.log10(power_magnitude)
    arr = np.array(power_db)

    # Reshape the array to the required shape (180, 450)
    arr_reshaped = arr.reshape(180, -1)  # -1 means calculate the size of this dimension

    df = pd.DataFrame(arr_reshaped)

    # Now df is a DataFrame with shape (180, 450)
    print(df.shape)

    df.head()
    df.to_csv(rf'C:\MasterThesis\v1.0\sub-{subject}\ses-001\{subject}_{run}_pupil_dataset.csv', index=False)

print(f"There are {how_many_nan} rows with NaN")
