# Basic Libraries
import pandas as pd                  # For data manipulation
import numpy as np                   # For numerical operations
import matplotlib.pyplot as plt      # For plotting graphs
import seaborn as sns                # For advanced data visualizations

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

