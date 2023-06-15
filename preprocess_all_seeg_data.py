import os

from pymef import MefSession
from bids import BIDSLayout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, decimate
from sklearn.decomposition import PCA
import mne
import eeglib

APPLY_LOW_PASS_FILTER = True
DIMENSION_REDUCTION = False
LSTM_READY = False
FEATURE_SET = 3
subject = '004'
run = 2

# Parameters
order = 3
# fs = target_freq       # sample rate, Hz
fs = 4800
cutoff = 50.0


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
    path_to_dataset = r"C:\MasterThesis\v1.0"  # Please change this value

    l = BIDSLayout(path_to_dataset)

    subject = srt[0]
    run = srt[1]
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

    # Get electrodes
    electrodes_file = l.get(suffix='electrodes', extension='tsv',
                            subject=json_entities['subject'],
                            session=json_entities['session'])[0]
    electrode_df = electrodes_file.get_df()
    electrode_df

    # Get channels
    channel_file = l.get(suffix='channels', extension='tsv',
                         task=json_entities['task'],
                         subject=json_entities['subject'],
                         session=json_entities['session'],
                         run=json_entities['run'])[0]
    channel_df = channel_file.get_df()
    channel_df = channel_df[channel_df['status'] != 'bad']

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

    regions = electrode_df['anatomy_structure']
    regions = set(regions)

    valid_electrodes = [x['name'] for idx, x in electrode_df.iterrows() if x['anatomy_structure'] in ['wm']]
    print(f'valid_electrodes = {valid_electrodes}')

    channels = valid_electrodes
    while len(channels) < 1:
        channels.extend(valid_electrodes)
    channels = channels[0:1]
    print(f'channels = {channels}')
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

    data_events = []
    for e in events_df['uutc']:
        data_events.append(ms.read_ts_channels_uutc(channels, [e - 700000, e + 1300000]))
    data_events = np.array(data_events)

    target_freq = 1000
    q = int(32000 / target_freq)
    downsampled_data_events = []

    for s_idx, sequence in enumerate(data_events):
        downsampled_sequence = []
        for ch_idx, channel in enumerate(sequence):
            downsampled_sequence.append(decimate(channel, q))
        downsampled_data_events.append(downsampled_sequence)

    data_events = np.array(downsampled_data_events)

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
    print(how_many_nan)

    channels_list = [[] for x in range(len(channels))]

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            channels_list[ch_idx].extend(channel)
    print(len(channels_list))

    for idx, channel in enumerate(channels_list):
        mean = np.mean(channel)
        std = np.std(channel)
        data_normalized = (channel - mean) / std
        channels_list[idx] = data_normalized

    samples_in_sequence = int(target_freq * 2.0)
    chunks_list = [[] for x in range(len(channels))]
    for idx, channel in enumerate(channels_list):
        chunks_list[idx] = split_list(channel, samples_in_sequence)

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            data_events[s_idx][ch_idx] = chunks_list[ch_idx][s_idx]

    data_events = np.array(data_events)

    if APPLY_LOW_PASS_FILTER:
        for s_idx, sequence in enumerate(data_events):
            for ch_idx, channel in enumerate(sequence):
                y = butter_lowpass_filter(channel, cutoff, fs, order)
                data_events[s_idx][ch_idx] = y

    if FEATURE_SET == 3:
        sfreq = target_freq  # replace with your sampling frequency
        # delta: 2.5-5 Hz, theta: 4–9 Hz, alpha: 9–16 Hz, beta: 16–25 Hz,
        freqs = [9., 16., 25.]  # define the range of frequency bands
        # freqs = [5.]
        print(freqs)
        n_cycles = 7.0

        # perform Morlet Wavelet transform
        power = mne.time_frequency.tfr_array_morlet(data_events, sfreq, freqs, n_cycles=n_cycles)
        # n_samples * n_channels * n_frequencies * n_times
        # power = 10 * np.log10(power)
    # power.shape
    # extract a power of the signal
    power_magnitude = np.abs(power)
    # convert to a decibel scale, which is often done for power spectral densities:
    power_db = 10 * np.log10(power_magnitude)

    arr = np.array(power_db)

    # Reshape the array to the required shape (180, 450)
    arr_reshaped = arr.reshape(180, -1)  # -1 means calculate the size of this dimension

    df = pd.DataFrame(arr_reshaped)

    # Now df is a DataFrame with shape (180, 450)
    print(df.shape)
    if DIMENSION_REDUCTION:
        pca = PCA(n_components=10)
        pca_data = pca.fit_transform(df)
        df = pd.DataFrame(pca_data)
        print(pca_data.shape)

    df.to_csv(rf'C:\MasterThesis\v1.0\sub-{subject}\ses-001\{subject}_{run}_sEEG_dataset_wm.csv', index=False)