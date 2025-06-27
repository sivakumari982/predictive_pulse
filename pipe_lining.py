import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_patient_data.csv")

# Features and Target
features = [
    "C", "Age", "History", "Patient", "TakeMedication", "Severity",
    "BreathShortness", "VisualChanges", "NoseBleeding",
    "Whendiagnoused", "ControlledDiet", "Systolic_Num", "Diastolic_Num"
]
X = df[features]
y = df["Stages"]

# Encode Target Labels (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for Flask app later
joblib.dump(label_encoder, "label_encoder.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model (SVM with best hyperparams)
final_model = make_pipeline(
    StandardScaler(),
    SVC(C=1, gamma=0.1, kernel='rbf', probability=True)
)

# Fit model
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "svm_classifier_pipeline.pkl")

print("✅ Model and label encoder saved successfully!")
import joblib

model = joblib.load("svm_classifier_pipeline.pkl")

# Example input: This should match your model's input feature format
input_data = [
    0,  # Gender (Male)
    0,  # Age Group (18-34)
    1,  # History
    1,  # Patient
    1,  # TakeMedication
    0,  # Severity (Mild)
    1,  # BreathShortness
    1,  # VisualChanges
    1,  # NoseBleeding
    1,  # Whendiagnosed
    1,  # ControlledDiet
    120,  # Systolic_Num
    80   # Diastolic_Num
]

pred = model.predict([input_data])[0]
print("Predicted Class:", pred)
