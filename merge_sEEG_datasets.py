import os
import pandas as pd

DATA_FOR_LSTM = False

directory = r"C:\MasterThesis\v1.0\\"
folders = os.listdir(directory)
subjects = []
list_of_dataframes = []
for f in folders:
    if 'sub' in f:
        subjects.append(f)


if not DATA_FOR_LSTM:
    for subject in subjects:
        # if 'sub-004' in subject:
        path = directory + subject + r'\ses-001\\'
        print(path)
        files = os.listdir(path)
        datasets = []
        labels = []
        for f in files:
            if 'sEEG_dataset' in f:
                datasets.append(f)
            elif 'labels' in f:
                labels.append(f)

        for d, l in zip(datasets, labels):
            print(f"dataset = {d}, label = {l}")
            df = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{d}")
            df_l = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{l}", sep='\t')
            df['labels'] = df_l['is_success']
            list_of_dataframes.append(df)
            # print(df)

    df = pd.concat(list_of_dataframes)
    df.to_csv(rf'C:\MasterThesis\v1.0\sEEG_dataset.csv', index=False)
