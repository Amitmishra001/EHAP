
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("Libraries are ready!")


uploaded = pickle.load(open('heart_disease_model.pkl', 'rb'))



# Load the dataset
df = pd.read_csv("heart.csv")

# Show the first 5 rows
df.head()

df.info()
df.isnull().sum()

df.columns

df.columns.tolist()

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Plot 1: Count of Heart Disease
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Heart Disease Distribution\n(0 = No, 1 = Yes)')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()

# Plot 2: Heart Disease by Sex
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='target', data=df, palette='Set1')
plt.title('Heart Disease by Gender (0 = Female, 1 = Male)')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Heart Disease')
plt.show()

# Plot 3: Chest Pain Type vs Heart Disease
plt.figure(figsize=(8,5))
sns.countplot(x='cp', hue='target', data=df, palette='Set3')
plt.title('Chest Pain Type and Heart Disease')
plt.xlabel('Chest Pain Type (0–3)')
plt.ylabel('Count')
plt.legend(title='Heart Disease')
plt.show()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Features")
plt.show()

X = df.drop('target', axis=1)  # X = all columns except target
y = df['target']               # y = only the target column (0 or 1)

X.head()

y.head()


df = pd.read_csv('heart.csv')
df.head()

X = df.drop('target', axis=1)  # X = all input columns
y = df['target']               # y = output column (0 = No heart disease, 1 = Yes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model using training data
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

import numpy as np

# Example person: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
sample_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# Reshape to match model input
sample_data = sample_data.reshape(1, -1)

# Make prediction
prediction = model.predict(sample_data)

# Show result
print("Prediction (1 = has heart disease, 0 = no):", prediction[0])

sample_df = pd.DataFrame([[
    52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3
]], columns=X.columns)

prediction = model.predict(sample_df)
print("Prediction (1 = has heart disease, 0 = no):", prediction[0])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test data
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display it nicely
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap='Blues')


from sklearn.metrics import classification_report

# Get predictions
y_pred_log = model.predict(X_test)

# Create the report
report_log = classification_report(y_test, y_pred_log, target_names=["No Disease", "Disease"])
print(report_log)

from sklearn.ensemble import RandomForestClassifier

# Create and train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Check accuracy
rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Model Accuracy:", rf_accuracy)

from sklearn.metrics import classification_report

# Use predictions from the Random Forest model
y_pred = rf_model.predict(X_test)

# Generate the report
report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
print(report)

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)
print("SVM Model Accuracy:", svm_accuracy)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print("KNN Model Accuracy:", knn_accuracy)



df = pd.read_csv('heart.csv')  # change this name if needed
df.head()

from sklearn.ensemble import RandomForestClassifier

# Create and train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Check accuracy
rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Model Accuracy:", rf_accuracy)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model using training data
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

import pickle

# Save the logistic regression model
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")



import pickle
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

import pickle

with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)


import pickle
import numpy as np

# Load the saved logistic regression model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample patient input — change these values to test different patients
# Format: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# Predict
prediction = model.predict(patient_data)

# Show result
if prediction[0] == 1:
    print("Prediction: YES – The patient has heart disease.")
else:
    print("Prediction: NO – The patient does not have heart disease.")

import pickle
import numpy as np

# Load the saved logistic regression model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample patient input — change these values to test different patients
# Format: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# Predict
prediction = model.predict(patient_data)

# Show result
if prediction[0] == 1:
    print("Prediction: YES – The patient has heart disease.")
else:
    print("Prediction: NO – The patient does not have heart disease.")

import pickle
import numpy as np

# Load the saved model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Collect user input
print("Enter patient details:")

age = int(input("Age: "))
sex = int(input("Sex (0 = Female, 1 = Male): "))
cp = int(input("Chest Pain Type (0–3): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar > 120? (1 = Yes, 0 = No): "))
restecg = int(input("Resting ECG Result (0–2): "))
thalach = int(input("Maximum Heart Rate Achieved: "))
exang = int(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
oldpeak = float(input("Oldpeak (ST depression): "))
slope = int(input("Slope of the ST segment (0–2): "))
ca = int(input("Number of Major Vessels (0–4): "))
thal = int(input("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

# Combine all inputs
patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]])

# Predict
prediction = model.predict(patient_data)

# Output
print("\n🔍 Prediction:")
if prediction[0] == 1:
    print("⚠️ YES – The patient has heart disease.")
else:
    print("✅ NO – The patient does NOT have heart disease.")

import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define patient data using a dictionary (edit values here)
patient_info = {
    "age": 58,
    "sex": 1,         # 1 = Male
    "cp": 2,          # Chest pain type
    "trestbps": 130,  # Resting BP
    "chol": 250,      # Cholesterol
    "fbs": 0,         # Fasting blood sugar > 120
    "restecg": 1,
    "thalach": 170,   # Max heart rate
    "exang": 0,       # Exercise induced angina
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2         # 1 = Normal, 2 = Fixed, 3 = Reversible
}

# Convert values to numpy array
patient_data = np.array([[value for value in patient_info.values()]])

# Make prediction
prediction = model.predict(patient_data)

# Show result
print("🔍 Patient Data:", patient_info)
print("✅ Prediction:", "YES – Heart Disease" if prediction[0] == 1 else "NO – No Heart Disease")

import pickle
import numpy as np

# Load the model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Patient data in order: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient_data = [60, 1, 0, 120, 200, 0, 1, 160, 0, 1.0, 1, 0, 2]

# Reshape and predict
patient_array = np.array(patient_data).reshape(1, -1)
prediction = model.predict(patient_array)

# Result
print("🩺 Prediction:", "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease")

import pickle
import numpy as np

# Load the model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Patient data in order: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient_data = [60, 1, 0, 120, 200, 0, 1, 160, 0, 1.0, 1, 0, 2]

# Reshape and predict
patient_array = np.array(patient_data).reshape(1, -1)
prediction = model.predict(patient_array)

# Result
print("🩺 Prediction:", "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease")

import pickle

# Assuming you have your trained model as `model`
with open('heart_disease_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)