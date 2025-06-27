import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Step 1: Read your CSV file
df = pd.read_csv("patient_data.csv")  # Replace with your actual filename

# Step 2: Clean BP columns (Systolic & Diastolic) to numerical values

def convert_bp(val):
    if pd.isnull(val):
        return None
    val = str(val).replace("+", "").strip()
    if "-" in val:
        parts = val.split("-")
        return (int(parts[0]) + int(parts[1])) // 2
    try:
        return int(val)
    except:
        return None

df["Systolic_Num"] = df["Systolic"].apply(convert_bp)
df["Diastolic_Num"] = df["Diastolic"].apply(convert_bp)

# Step 3: Encode categorical columns (Yes/No, etc.)
df_clean = df.copy()

# Exclude target and BP columns
categorical_cols = [
    "C", "Age", "History", "Patient", "TakeMedication", "Severity",
    "BreathShortness", "VisualChanges", "NoseBleeding",
    "Whendiagnoused", "ControlledDiet"
]

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Step 4: Prepare feature matrix X and target variable y
X = df_clean[categorical_cols + ["Systolic_Num", "Diastolic_Num"]]
X = X.fillna(X.mean())  # handle any missing values

# Target variable
y = df["Stages"]
y_encoded = LabelEncoder().fit_transform(y)


# Step 5: Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM":SVC()
}

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Cross-validation results (accuracy):")
for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipeline, X, y_encoded, cv=skf, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.3f} | Std = {scores.std():.3f}")
