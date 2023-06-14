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
        if 'sub-004' in subject:
            path = directory + subject + r'\ses-001\\'
            print(path)
            files = os.listdir(path)
            seeg_datasets = []
            pupil_datasets = []
            labels = []
            for f in files:
                if 'sEEG_dataset' in f:
                    seeg_datasets.append(f)
                elif 'pupil_dataset' in f:
                    pupil_datasets.append(f)
                elif 'labels' in f:
                    labels.append(f)

            for s, p, l in zip(seeg_datasets, pupil_datasets, labels):
                print(f"seeg_datasets = {s}, pupil_datasets = {p}, label = {l}")
                df_s = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{s}")
                df_p = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{p}")
                df_l = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{l}", sep='\t')
                df = pd.concat([df_s, df_p], axis=1)
                df['labels'] = df_l['is_success']
                list_of_dataframes.append(df)
                # print(df)

    df = pd.concat(list_of_dataframes)
    df.to_csv(rf'C:\MasterThesis\v1.0\multi_dataset.csv', index=False)
