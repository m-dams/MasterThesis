import os
import pandas as pd

directory = r"C:\MasterThesis\v1.0\\"
final_dataset = pd.DataFrame(columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'labels'])
folders = os.listdir(directory)
subjects = []
for f in folders:
    if 'sub' in f:
        subjects.append(f)

for subject in subjects:
    path = directory + subject + r'\ses-001\\'
    print(path)
    files = os.listdir(path)
    datasets = []
    labels = []
    for f in files:
        if 'dataset' in f:
            datasets.append(f)
        elif 'labels' in f:
            labels.append(f)
    for d, l in zip(datasets, labels):
        df = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{d}")
        df_l = pd.read_csv(rf"C:\MasterThesis\v1.0\{subject}\ses-001\{l}", sep='\t')
        df['labels'] = df_l['is_success']
        # print(df)
        final_dataset.append(df, ignore_index=False, verify_integrity=False)
        print(final_dataset)

final_dataset.to_csv(rf'C:\MasterThesis\v1.0\pupil_dataset.csv', index=False)
