# Predictive Pulse – Blood Pressure Stage Predictor 🚑

**Predictive Pulse** is a machine learning-based application that analyzes health data (like blood pressure, glucose, age, BMI) and predicts the **blood pressure stage** of a patient.

## 🧠 Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Flask (for web interface)
- Machine Learning models: Logistic Regression, Random Forest, Gradient Boosting
- Data Visualization
- HTML/CSS (basic UI)

## 🔍 Features
- Predicts BP stage based on user inputs
- Compares accuracy of multiple ML models using cross-validation
- Result-based health advice (e.g., "See a doctor", "Maintain diet")
- Generates prediction summary PDF (Flask backend)
- Visual chart for user-entered values



## 🚀 How to Run Locally
```bash
git clone https://github.com/sivakumari982/predictive_pulse
cd predictive_pulse
pip install -r requirements.txt
python app.py
