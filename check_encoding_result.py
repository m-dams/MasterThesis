import pandas as pd
import os


def add_encoding_success_column(filename, file_path):
    data = pd.read_csv(file_path, sep='\t')
    print(data.head())
    recalled_words = []
    df = pd.DataFrame(columns=["onset", "duration", "sample", "trial_type", "value", "text", "word_id", "list", "is_success"])

    for index, row in data.iterrows():
        if row['trial_type'] == 'RECALL':
            recalled_words.append(row['text'])

    for index, row in data.iterrows():
        if row['trial_type'] == 'ENCODE' and row['text'] in recalled_words:
            row['word_id'] = int(row['word_id'])
            row['list'] = int(row['list'])
            row['is_success'] = 1
            df = df.append(row, ignore_index=True)
        elif row['trial_type'] == 'ENCODE' and row['text'] not in recalled_words:
            row['word_id'] = int(row['word_id'])
            row['list'] = int(row['list'])
            row['is_success'] = 0
            df = df.append(row, ignore_index=True)

    print(df)
    save_path = save_dir + filename[4:7] + filename[27:30]
    save_path += '_labels.tsv'
    print(f"save path = {save_path}")
    df.to_csv(save_path, sep='\t', index=False)


subject = '12'
directory = rf'C:\MasterThesis\v1.0\sub-0{subject}\ses-001\ieeg\\'
save_dir = rf'C:\MasterThesis\v1.0\sub-0{subject}\ses-001\\'
# file_path = r'C:\MasterThesis\v1.0\sub-004\ses-001\ieeg\sub-004_ses-001_task-FR_run-01_events.tsv'
# save_path = r'C:\MasterThesis\v1.0\sub-004\ses-001\ieeg\sub-004_ses-001_task-FR_run-01_events_encoding.tsv'

for filename in os.listdir(directory):
    if 'tsv' in filename:
        if 'events' in filename:
            if 'FR' in filename:
                print(filename)
                add_encoding_success_column(filename, directory + filename)
