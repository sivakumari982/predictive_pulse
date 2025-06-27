import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load dataset
df = pd.read_csv("patient_data.csv")

# Fix typos in 'Stages'
df["Stages"] = df["Stages"].replace({
    "HYPERTENSION (Stage-2).": "HYPERTENSION (Stage-2)",
    "HYPERTENSIVE CRISI": "HYPERTENSIVE CRISIS"
})

# Convert BP strings to average integers
def convert_bp(val):
    if pd.isnull(val):
        return None
    val = str(val).replace("+", "").strip()
    if "-" in val:
        try:
            parts = val.split("-")
            return (int(parts[0]) + int(parts[1])) // 2
        except:
            return None
    try:
        return int(val)
    except:
        return None

df["Systolic_Num"] = df["Systolic"].apply(convert_bp)
df["Diastolic_Num"] = df["Diastolic"].apply(convert_bp)

# Encode categorical features
df_clean = df.copy()
categorical_cols = [
    "C", "Age", "History", "Patient", "TakeMedication", "Severity",
    "BreathShortness", "VisualChanges", "NoseBleeding", "Whendiagnoused", "ControlledDiet"
]

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Select final features (including Age)
important_features = [
    "History", "Patient", "TakeMedication", "Severity",
    "BreathShortness", "VisualChanges", "NoseBleeding",
    "ControlledDiet", "Systolic_Num", "Diastolic_Num", "Age"
]

X = df_clean[important_features].fillna(0)

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Stages"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM model
model = make_pipeline(StandardScaler(), SVC(C=1, gamma=0.1, kernel='rbf'))
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "svm_classifier_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Print result
print("Stages")
print(df["Stages"].value_counts())
print("✅ Model and label encoder saved successfully!")
