
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/AmenaNajeeb/Data/refs/heads/master/autism_screening.csv')
df = df.rename(columns={"austim": "autism"})

df['gender'] = df['gender'].str.lower().str.strip().replace({'m': 'male', 'f': 'female'})
df['jundice'] = df['jundice'].str.lower().str.strip().replace({'y': 'yes', 'n': 'no'})
df['autism'] = df['autism'].str.lower().str.strip().replace({'y': 'yes', 'n': 'no'})
df['contry_of_res'] = df['contry_of_res'].str.lower().str.strip()
df['ethnicity'] = df['ethnicity'].str.lower().str.strip()


# Drop unused or null columns
df['age'] = df['age'].fillna(df['age'].mean())
df = df.drop(columns=['age_desc', 'used_app_before', 'relation'], errors='ignore')
df = df.rename(columns={"Class/ASD": "ASD_Class"})

# Encode object columns
label_encoders = {}
for col in ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    

# Split
X = df.drop('ASD_Class', axis=1)
y = df['ASD_Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Train model
model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)