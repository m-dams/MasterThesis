import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Odczytanie danych z plików .pkl (zakładając, że mamy listę ścieżek do plików .pkl)
# file_paths = ['path/to/file1.pkl', 'path/to/file2.pkl', ...]
# data_frames = [pd.read_pickle(file_path) for file_path in file_paths]


file_path = r"C:\MasterThesis\v1.0\derivatives\sub-004_ses-001_run-01_task-AP.pkl"

# Otwórz plik w trybie odczytu binarnego (rb)
with open(file_path, 'rb') as file:
    # Odczytaj dane z pliku
    data = pickle.load(file)

data = pd.DataFrame(data)
for col in data.columns:
    print(col)
# data = pd.concat(data_frames)
print(data.head())
print(data.shape)
print(data.columns)
print(data.iloc[8291504])




# # Przetwarzanie danych w celu wyodrębnienia cech i etykiet (przykład może wymagać modyfikacji)
# X = data[['amplitude', 'duration', 'frequency']]  # cechy
# y = data['event_class']  # etykiety
#
# # Podział danych na zbiory treningowe i testowe
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Normalizacja cech
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Wybór modelu uczenia maszynowego i trening
# clf = SVC(kernel='linear', C=1)
# clf.fit(X_train_scaled, y_train)
#
# # Ewaluacja modelu
# y_pred = clf.predict(X_test_scaled)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
