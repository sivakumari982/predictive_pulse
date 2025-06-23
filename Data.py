# Basic Libraries
import pandas as pd                  # For data manipulation
import numpy as np                   # For numerical operations
import matplotlib.pyplot as plt      # For plotting graphs
import seaborn as sns 
import matplotlib   
            # For advanced data visualizations
            # For advanced data visualizations

# Preprocessing & Utilities
import warnings                      # To ignore warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split     # For splitting data
from sklearn.preprocessing import LabelEncoder           # Encoding categorical features
from sklearn.preprocessing import StandardScaler         # Normalization

# Machine Learning Models
from sklearn.svm import SVC                              # Support Vector Machine
from sklearn.linear_model import LogisticRegression      # Logistic Regression
from sklearn.ensemble import RandomForestClassifier      # Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier       # KNN Classifier
from sklearn.tree import DecisionTreeClassifier          # Decision Tree Classifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model Saving
import joblib                                             # To save and load trained models


df = pd.read_csv('patient_data.csv')

print(df.isnull())

print(df.isnull().sum())

df = df.drop_duplicates()

print(df.duplicated().value_counts())
print(df.dtypes)

for col in df.columns:
    # Remove spaces and symbols from values
    df[col] = df[col].astype(str).str.strip().str.replace(r'[^\d.]', '', regex=True)

    # Try converting to numeric (if it fails, it will remain object)
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Step 4: Show data types after conversion
print("\nUpdated Data Types:")
print(df.dtypes)
print("Actual Columns (with indexes):")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")
df.columns = df.columns.str.strip()         # Remove spaces
df.columns = df.columns.str.replace(' ', '') # Remove inner spaces
print("Diastolic" in df.columns)  # This should print: True








# Step 5: Now describe numeric values
print("\nDescriptive Statistics (with quartiles):")
print(df.describe())


sns.countplot(x='Stages', data=df)
plt.title("Activity Level Distribution")
plt.show()



df1 = pd.DataFrame({
    'ControlledDiet': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Strip spaces if needed
df1['ControlledDiet'] = df1['ControlledDiet'].astype(str).str.strip()

# Pie Chart
df1['ControlledDiet'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Controlled Diet Status")
plt.ylabel("")  # Remove the default y-label
plt.show()

sns.histplot(df['Diastolic'], kde=True)
plt.title("Distribution of Systolic Blood Pressure")
plt.show()

sns.barplot(x='Stages', y='Systolic', data=df)
plt.title("Average Systolic BP by Health Stage")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, square=True)
plt.title("Correlation Heatmap")
plt.show()

