import os
from pymef import MefSession
from bids import BIDSLayout
import scipy.fft
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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

    # Get channels
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
    events_df

    # Since we are using MEF3 pybids has problems recognizing the extension so we need to modify the json extension
    mef_session_path = os.path.splitext(json_file.path)[0] + '.mefd'
    print(mef_session_path)
    ms = MefSession(mef_session_path, None)
    ms.read_ts_channel_basic_info()

    channels = ['LEFT_PUPIL_SIZE', 'RIGHT_PUPIL_SIZE']

    start_time = 1553941347170839 + 3 * 1e6  # in microseconds
    end_time = 1553942904095135 - 3 * 1e6
    win_size = 10  # in seconds
    stop_time = start_time + int(win_size * 1e6)

    bi = ms.read_ts_channel_basic_info()

    records = ms.read_records()
    records_in_win = [x for x in records if start_time < x['time'] < stop_time]
    session_metadata = ms.session_md
    session_start_utc = session_metadata['session_specific_metadata']['earliest_start_time'][0]
    print(f"session_start_utc = {session_start_utc}\nsession_stop_utc = {stop_time}")
    events_df['microsecond_onset'] = events_df['onset'] * 1e6
    events_df['uutc'] = (events_df['microsecond_onset'] + session_start_utc).astype('int64')
    events_in_win = events_df[(start_time < events_df['uutc'])
                              & (events_df['uutc'] < stop_time)]
    events_df.head()

    records_in_events = []
    for x in records:
        for e in events_df['uutc']:
            if 0 == (x['time'] - e):
                records_in_events.append(x)

    len(records_in_events)
    data_events = []
    for e in events_df['uutc']:
        data_events.append(ms.read_ts_channels_uutc(channels, [e - 500000, e + 1000000]))
    len(data_events)

    print(data_events[0])

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            if check_missing_values(channel):
                how_many_nan += 1
                df = pd.DataFrame({'signal': channel})
                print(f"Apply linear interpolation to fill the missing values [{s_idx}][{ch_idx}]")
                df['signal'] = df['signal'].bfill().ffill().interpolate(method='linear')
                data_events[s_idx][ch_idx] = df['signal'].values.tolist()
                print(data_events[s_idx][ch_idx])

    order = 3
    fs = 150.0  # sample rate, Hz
    cutoff = 30.0  # desired cutoff frequency of the filter, Hz

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            mean_pupil_size = np.mean(channel)
            std_pupil_size = np.std(channel)

            # Normalizacja z-score
            normalized_pupil_sizes = [(x - mean_pupil_size) / std_pupil_size for x in channel]
            print(normalized_pupil_sizes)
            data_events[s_idx][ch_idx] = normalized_pupil_sizes

    for s_idx, sequence in enumerate(data_events):
        for ch_idx, channel in enumerate(sequence):
            eeg_data = channel
            mean = np.mean(eeg_data)
            variance = np.var(eeg_data)
            std_dev = np.std(eeg_data)
            power_spectrum = np.abs(scipy.fft.fft(eeg_data)) ** 2
            freqs = scipy.fft.fftfreq(len(eeg_data), 1 / 128)
            delta_power = np.mean(power_spectrum[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.mean(power_spectrum[(freqs > 4) & (freqs <= 8)])
            alpha_power = np.mean(power_spectrum[(freqs > 8) & (freqs <= 12)])
            beta_power = np.mean(power_spectrum[(freqs > 12) & (freqs <= 30)])

            features = [mean, variance, std_dev, delta_power, theta_power, alpha_power, beta_power]

            print("Extracted features:", features)
            data_events[s_idx][ch_idx] = features

    arr = np.array(data_events)
    arr_reshaped = arr.reshape(180, -1)  # -1 means calculate the size of this dimension
    df = pd.DataFrame(arr_reshaped)
    print(df.shape)

    df.to_csv(rf'C:\MasterThesis\v1.0\sub-{subject}\ses-001\{subject}_{run}_pupil_dataset.csv', index=False)
print(f"There are {how_many_nan} rows with NaN")
