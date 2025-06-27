import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("patient_data.csv")

# Handle missing values
# Fill categorical columns with 'Missing'
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna("Missing")

# Fill numeric columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Label encode all columns
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Stages", axis=1)
y = df["Stages"].apply(lambda x: 0 if x == 0 else 1)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neural_network import MLPClassifier

# Define and train ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_ann = ann_model.predict(X_test)
print("ANN Accuracy:", accuracy_score(y_test, y_pred_ann))
print("\nANN Classification Report:\n", classification_report(y_test, y_pred_ann))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ann)
print(cm)

# Binary metrics
precision = precision_score(y_test, y_pred_ann)
recall = recall_score(y_test, y_pred_ann)
f1 = f1_score(y_test, y_pred_ann)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")


