import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. DATA LOADING & SMART CLEANING
print("🔄 Dataset load aur targeted cleaning ho rha hai...")
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Step A: Saare column names ke aage-piche ke spaces saaf karo
df.columns = df.columns.str.strip()

# Step B: Har text column ki entries ke aage-piche ke spaces saaf karo
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Step C: Age column ko target karein. Jo row mein text ('Negative') hai, use NaN banayein aur drop karein
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])

# Step D: Only keep diseases with 5 or more cases (Imbalanced Data Fix)
disease_counts = df['Disease'].value_counts()
valid_diseases = disease_counts[disease_counts >= 5].index
df_filtered = df[df['Disease'].isin(valid_diseases)].copy()

# Features (X) and Target Label (Y)
Y = df_filtered['Disease']
X_raw = df_filtered.drop(columns=['Disease'])

# Step E: Automatic One-Hot Encoding for text categories
X_encoded = pd.get_dummies(X_raw, drop_first=False)

# Train-Test Split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.20, random_state=42)

# 2. TRAIN THE MODEL
print("🧠 AI Model train ho rha hai...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Show Baseline Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(f"✅ Success! Model Accuracy: {acc * 100:.2f}%\n")

# 3. INTERACTIVE INTERFACE
print("="*40)
print("🩺 LIVE CLINIC PREDICTOR INTERFACE 🩺")
print("="*40)
print("Patient ki details enter kijiye:\n")

fever = input("Does the patient have a Fever? (Yes/No): ").strip()
cough = input("Does the patient have a Cough? (Yes/No): ").strip()
fatigue = input("Does the patient have Fatigue? (Yes/No): ").strip()
breathing = input("Difficulty Breathing? (Yes/No): ").strip()
age = float(input("Patient Age (e.g., 25): ").strip())
gender = input("Gender (Male/Female): ").strip()
bp = input("Blood Pressure (High/Normal/Low): ").strip()
chol = input("Cholesterol Level (High/Normal): ").strip()

# Input processing
custom_patient = pd.DataFrame([{
    'Fever': fever, 'Cough': cough, 'Fatigue': fatigue, 
    'Difficulty Breathing': breathing, 'Age': age, 'Gender': gender, 
    'Blood Pressure': bp, 'Cholesterol Level': chol
}])

custom_encoded = pd.get_dummies(custom_patient)
custom_encoded = custom_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Final Prediction
predicted_disease = model.predict(custom_encoded)[0]

print("\n" + "-"*40)
print(f"🤖 AI DIAGNOSIS: The model predicts **{predicted_disease}**")
print("-"*40)