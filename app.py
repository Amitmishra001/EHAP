from flask import Flask, render_template, request, jsonify
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the pre-trained model
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

    df = pd.read_csv("heart.csv")
    # Show the first 5 rows
df.head()
df.info()
df.isnull().sum()
df.columns
df.columns.tolist()


X = df.drop('target', axis=1)  # X = all columns except target
y = df['target']               # y = only the target column (0 or 1)

X.head()
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load your data
data = pd.read_csv('heart.csv')

# Select only 6 features
X = data[['age', 'sex', 'trestbps', 'chol', 'cp', 'thalach']]
y = data['target']

# Train a new Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the new model
joblib.dump(model, 'heart_disease_model.pkl')

from fpdf import FPDF
import os

import os

# Get absolute path for saving files
from fpdf import FPDF
import os

# Ensure the static/reports directory exists
report_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'reports')
os.makedirs(report_dir, exist_ok=True)

# Define PDF Class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Heart Disease Prediction Report', ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln(5)

# Middleware Function
def generate_pdf_report(name,age, sex, bp, chol, cp, thalach, prediction_result, suggestion_text):
    # Format patient data
    patient_data = (
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Sex: {'Male' if sex == 1 else 'Female'}\n"
        f"Blood Pressure: {bp} mmHg\n"
        f"Cholesterol: {chol} mg/dL\n"
        f"Chest Pain Type: {cp}\n"
        f"Max Heart Rate (thalach): {thalach} bpm"
    )

    # Format prediction
    prediction_info = f"Prediction: {prediction_result}"

    # Now Create PDF
    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title("Patient Details:")
    pdf.chapter_body(patient_data)

    pdf.chapter_title("Prediction Result:")
    pdf.chapter_body(prediction_info)

    pdf.chapter_title("Suggestions:")
    pdf.chapter_body(suggestion_text)

    # Define output file path
    pdf_output_path = os.path.join(report_dir, 'patient_report.pdf')

    # Check if file exists, and remove it if necessary
    if os.path.exists(pdf_output_path):
        os.remove(pdf_output_path)

    # Save PDF (this will overwrite the existing file if it exists)
    pdf.output(pdf_output_path)

    return pdf_output_path


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request for prediction")
        # Get data from the request
        name = request.form['name']
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bp = float(request.form['bp'])
        chol = float(request.form['chol'])
        cp = float(request.form['cp'])
        thalach = float(request.form['thalach'])
        
        features = [age, sex, cp, bp, chol, thalach]  # Adjust order if needed
        prediction = model.predict([features])[0]
        
        prediction_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        # Suggestion based on prediction
        suggestion = ("Consult a doctor immediately and adopt a healthy lifestyle."
                      if prediction == 1 else
                      "Maintain your healthy lifestyle. Regular checkups are recommended.")
        
        # Generate PDF Report
        pdf_path = generate_pdf_report(name,age, sex, bp, chol, cp, thalach, prediction_text, suggestion)
        pdf_url = 'http://127.0.0.1:5000/static/reports/patient_report.pdf'
        return render_template('index.html', prediction_text=prediction_text, report_link=pdf_path,pdf_url=pdf_url)
                

        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)