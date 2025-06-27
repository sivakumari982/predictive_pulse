from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("svm_classifier_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Use the same 11 features used during training
        input_data = [
            int(request.form.get('History')),
            int(request.form.get('Patient')),
            int(request.form.get('TakeMedication')),
            int(request.form.get('Severity')),
            int(request.form.get('BreathShortness')),
            int(request.form.get('VisualChanges')),
            int(request.form.get('NoseBleeding')),
            int(request.form.get('ControlledDiet')),
            int(request.form.get('Systolic_Num')),
            int(request.form.get('Diastolic_Num')),
            int(request.form.get('Age'))  # Make sure 'Age' is encoded like in training
        ]

        prediction = model.predict([input_data])[0]
        decoded_prediction = label_encoder.inverse_transform([prediction])[0]
        # Advice based on prediction
        advice_map = {
            "NORMAL": "Your blood pressure is normal. Maintain a healthy lifestyle!",
            "HYPERTENSION (Stage-1)": "Monitor your BP regularly and reduce salt intake.",
            "HYPERTENSION (Stage-2)": "Consult a doctor and follow medication strictly.",
            "HYPERTENSIVE CRISIS": "Immediate medical attention is required!"
        }
        advice = advice_map.get(decoded_prediction, "Please consult a healthcare provider.")
        return render_template(
            'index.html',
            prediction_text=f"Predicted Hypertension Stage: {decoded_prediction}",
            advice=advice,
            systolic=input_data[-3],  # Systolic_Num
            diastolic=input_data[-2]  # Diastolic_Num
        )
    except Exception as e:
        return f"Error: {e}"
if __name__ == '__main__':
    app.run(debug=True)
