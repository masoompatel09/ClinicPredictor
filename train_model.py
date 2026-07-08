import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. PREPARE THE DATASET (From Previous Steps)
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")
disease_counts = df['Disease'].value_counts()
valid_diseases = disease_counts[disease_counts >= 5].index
df_filtered = df[df['Disease'].isin(valid_diseases)].copy()

Y = df_filtered['Disease']
X_raw = df_filtered.drop(columns=['Disease'])
X_encoded = pd.get_dummies(X_raw, columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level'])

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.20, random_state=42)

print("--- Step 7: Training the Random Forest Classifier ---")

# 2. INITIALIZE AND TRAIN THE MODEL
# n_estimators=100 means we are building a forest of 100 individual decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# 3. EVALUATE ON THE TEST DATA (The 50 hidden patients)
predictions = model.predict(X_test)

# 4. PRINT PERFORMANCE RESULTS
accuracy = accuracy_score(Y_test, predictions)
print(f"\n🎯 Model Overall Accuracy: {accuracy * 100:.2f}%\n")

print("--- Detailed Performance Report per Disease ---")
print(classification_report(Y_test, predictions, zero_division=0))