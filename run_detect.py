#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:49:50 2019

Script to run EPYCOM algorithms on memory encoding dataset

Ing.,Mgr. (MSc.) Jan Cimbálník
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

import os, argparse
from time import time

import numpy as np
import pandas as pd

from bids import BIDSLayout
from epycom.event_detection import HilbertDetector
from pymef import MefSession

from pathlib import Path

# file_path = Path(__file__).parent.absolute()
file_path = r'C:\MasterThesis\v1.0'

# %% Parse arguments

parser = argparse.ArgumentParser(description='Memory encoding task detection of HFO')

parser.add_argument('--subjects', default=None, action="store",
                    nargs='+', type=int, help='Specify BIDS subject')
parser.add_argument('--sessions', default=None, action="store",
                    nargs='+', type=int, help='Specify BIDS session')
parser.add_argument('--tasks', default=None, action="store",
                    nargs='+', type=str, help='Specify BIDS task')
parser.add_argument('--runs', default=None, action="store",
                    nargs='+', type=int, help='Specify BIDS run')

parser.add_argument('--channels', default=None, action="store",
                    nargs='+', type=str, help='Specify iEEG channels run')

parser.add_argument('--n_cores', default=None, action="store", type=int,
                    help='Specify number of cores for multiprocessing')

parsed_vals = parser.parse_args()

subjects = parsed_vals.subjects
sessions = parsed_vals.sessions
tasks = parsed_vals.tasks
runs = parsed_vals.runs

channels = parsed_vals.channels

n_cores = parsed_vals.n_cores

# %% Presets

fs = 32000  # We know this for iEEG

# path_to_dataset = str(file_path.parent) + '/'
path_to_dataset = file_path + '/'
path_to_source = path_to_dataset + 'sourcedata/'
path_to_derivatives = path_to_dataset + 'derivatives/'
os.makedirs(path_to_derivatives, exist_ok=True)

mef_pwd = None

pre_offset = 1.25  # in seconds
post_offset = 1.25  # in seconds

compute_instance = HilbertDetector(low_fc=60,
                                   high_fc=800,
                                   band_spacing='log',
                                   num_bands=300,
                                   cyc_th=1,
                                   threshold=2)

# %% Processing - in small windows (2.5 seconds)
"""
The middle point for the window is:
COUNTDOWN - appearance of the countdown digit
ENCODING - appearance of the word(s)
DISTRACTOR - appearance of the equation
RECALL - start of word vocalization
"""

# Get data files
l = BIDSLayout(path_to_dataset)
filter_dict = {'suffix': 'ieeg',
               'extension': 'json'}
if subjects is not None:
    filter_dict['subject'] = [str(x).zfill(3) for x in subjects]
if sessions is not None:
    filter_dict['session'] = [str(x).zfill(3) for x in sessions]
if tasks is not None:
    filter_dict['task'] = tasks
if runs is not None:
    filter_dict['run'] = runs
json_files = l.get(**filter_dict)

mef_sessions = []
channel_dfs = []
event_dfs = []
for json_file in json_files:
    json_entities = json_file.entities

    # Get valid channels
    channel_file = l.get(suffix='channels', extension='tsv',
                         task=json_entities['task'],
                         subject=json_entities['subject'],
                         session=json_entities['session'],
                         run=json_entities['run'])[0]
    channel_df = channel_file.get_df()
    channel_dfs.append(channel_df[channel_df['type'].isin(['SEEG', 'ECOG'])])

    # Get events of interest
    events_file = l.get(suffix='events', extension='tsv',
                        task=json_entities['task'],
                        subject=json_entities['subject'],
                        session=json_entities['session'],
                        run=json_entities['run'])[0]
    events_df = events_file.get_df()
    # events_df = events_df.loc[~events_df['trial_type'].isna(), ['onset', 'sample', 'duration', 'trial_type', 'list']]
    events_df = events_df.loc[~events_df['trial_type'].isna(), ['onset', 'sample', 'duration', 'trial_type']]
    # During recall we want to analyze the start of vocalization - we need to correct for that in PAL task
    if json_entities['task'] == 'PAL':
        events_df.loc[events_df['trial_type'] == 'RECALL', 'sample'] += np.floor(
            events_df.loc[events_df['trial_type'] == 'RECALL', 'duration'] * fs).astype(int)
        events_df.loc[events_df['trial_type'] == 'RECALL', 'onset'] += events_df.loc[
            events_df['trial_type'] == 'RECALL', 'duration']
    event_dfs.append(events_df.loc[:, ['onset', 'sample', 'trial_type']])

    mef_session_path = os.path.splitext(json_file.path)[0] + '.mefd'
    mef_sessions.append(MefSession(mef_session_path, mef_pwd))

for ms, ch_df, ev_df, json_file in zip(mef_sessions, channel_dfs, event_dfs, json_files):

    print(f"Working on {json_file.filename[:-5]}")

    json_entities = json_file.entities

    # Get uUTC unique times (PAL task can have the times doubled)
    uq_times = ev_df['onset'].unique()
    uq_times *= 1e6
    uq_uutc_times = (uq_times + ms.session_md['session_specific_metadata']['earliest_start_time'][0]).astype(int)
    uq_samples = ev_df['sample'].unique()

    # Basic channel info
    bi = ms.read_ts_channel_basic_info()

    # Iterate over channels
    all_chan_df_list = []
    for ch in list(ch_df['name']):

        if channels is not None:
            if ch not in channels:
                continue

        print(f"\tDetecting channel {ch}")

        data = ms.read_ts_channels_sample(ch, [None, None])
        fs = [x['fsamp'] for x in bi if x['name'] == ch][0][0]
        ch_start_time = [x['start_time'] for x in bi if x['name'] == ch][0][0]
        compute_instance.params['fs'] = fs

        # Samp offsets
        pre_samp_offset = int(pre_offset * fs)
        post_samp_offset = int(post_offset * fs)

        # Since our window is constant we can create "artifical" channel by gluing the segments together
        starts = [x - pre_samp_offset for x in uq_samples]
        stops = [x + post_samp_offset for x in uq_samples]
        idx_arr = np.concatenate([np.arange(x, y) for x, y in zip(starts, stops)])

        seg_data = data[idx_arr]

        t = time()

        # Run windowed function over the segmented data
        res = compute_instance.run_windowed(seg_data,
                                            window_size=pre_samp_offset + post_samp_offset,
                                            n_cores=n_cores)
        res_df = pd.DataFrame(res)

        # Correct the window starts and stops based on sample starts/stops
        for win_idx in res_df.win_idx.unique():
            win_start = starts[win_idx]
            res_df.loc[res_df['win_idx'] == win_idx, ['event_start', 'event_stop']] += win_start

        res_df['event_start'] = (((res_df['event_start'] / fs) * 1e6) + ch_start_time).astype(np.int)
        res_df['event_stop'] = (((res_df['event_stop'] / fs) * 1e6) + ch_start_time).astype(np.int)

        # Assign channel
        res_df['channel'] = ch

        print(f"\tProcessed in {time() - t} s")
        all_chan_df_list.append(res_df)

    # Save the dataframe (on disk for now)
    all_chan_df = pd.concat(all_chan_df_list)
    sub = json_entities['subject']
    ses = json_entities['session']
    run = json_entities['run']
    task = json_entities['task']
    all_chan_df.to_pickle(f"{path_to_derivatives}sub-{sub}_ses-{ses}_run-{run}_task-{task}.pkl")
