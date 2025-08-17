import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/AmenaNajeeb/Data/refs/heads/master/autism_screening.csv')
df = df.rename(columns={"austim": "autism", "Class/ASD": "ASD_Class"})

df['gender'] = df['gender'].str.lower().str.strip().replace({'m': 'male', 'f': 'female'})
df['jundice'] = df['jundice'].str.lower().str.strip().replace({'y': 'yes', 'n': 'no'})
df['autism'] = df['autism'].str.lower().str.strip().replace({'y': 'yes', 'n': 'no'})
df['contry_of_res'] = df['contry_of_res'].str.lower().str.strip()
df['ethnicity'] = df['ethnicity'].str.lower().str.strip()
df['age'] = df['age'].fillna(df['age'].mean())

df = df.drop(columns=['result','age_desc', 'used_app_before', 'relation'], errors='ignore')
df['ethnicity'] = df['ethnicity'].replace({'?': 'white',})

categorical_fields = ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']

label_encoders = {}
for field in categorical_fields:
    le = LabelEncoder()
    df[field] = le.fit_transform(df[field])
    label_encoders[field] = le

df['ASD_Class'] = df['ASD_Class'].str.lower().map({'yes': 1, 'no': 0})
X = df.drop(['ASD_Class'], axis=1)
y = df['ASD_Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(" Classification report:\n", classification_report(y_test, y_pred))
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
