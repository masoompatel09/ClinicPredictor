from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- 1. MODEL TRAIN KARNA (BACKEND STARTUP) ---
print("🔄 Dataset load aur clean ho rha hai...")
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Standard targeted cleaning
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])

disease_counts = df['Disease'].value_counts()
valid_diseases = disease_counts[disease_counts >= 5].index
df_filtered = df[df['Disease'].isin(valid_diseases)].copy()

Y = df_filtered['Disease']
X_raw = df_filtered.drop(columns=['Disease'])
X_encoded = pd.get_dummies(X_raw, drop_first=False)

print("🧠 AI Model train ho rha hai...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, Y)
print("✅ Server aur AI Model tayar hain!")

# --- 2. WEB APP ROUTES ---

# Home Page Route: Jahan Khali Form Dikhega
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route: Jab Doctor 'Predict' Button Dabayega
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # HTML Form se saara data nikalna
        fever = request.form['fever']
        cough = request.form['cough']
        fatigue = request.form['fatigue']
        breathing = request.form['breathing']
        age = float(request.form['age'])
        gender = request.form['gender']
        bp = request.form['bp']
        chol = request.form['chol']
        
        # User input ko training format ke mutabik dataframe mein badalna
        custom_patient = pd.DataFrame([{
            'Fever': fever, 'Cough': cough, 'Fatigue': fatigue, 
            'Difficulty Breathing': breathing, 'Age': age, 'Gender': gender, 
            'Blood Pressure': bp, 'Cholesterol Level': chol
        }])
        
        # Encoding aur alignment
        custom_encoded = pd.get_dummies(custom_patient)
        custom_encoded = custom_encoded.reindex(columns=X_encoded.columns, fill_value=0)
        
        # Live AI Diagnosis
        predicted_disease = model.predict(custom_encoded)[0]
        
        # Result ke saath page ko wapas render karna
        return render_template('index.html', prediction=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)