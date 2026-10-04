import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# STEP 1: LOAD AND UNDERSTAND THE DATA
# ==========================================

df = pd.read_csv(BASE_DIR / "train.csv")

# print(df.head())

# print("\nShape of Dataset:")
# print(df.shape)

# print("\nDataset Information:")
# df.info()

# print("\nStatistical Summary:")
# print(df.describe())


# ==========================================
# STEP 2: CHECK MISSING VALUES
# ==========================================

# print("\nMissing Values:")
# print(df.isnull().sum())


# ==========================================
# STEP 3: HANDLE MISSING VALUES
# ==========================================

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#Remove Cabin because it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Verify missing values after cleaning
# print("\nMissing Values After Cleaning:")
# print(df.isnull().sum())


# ==========================================
# STEP 4: CHECK DUPLICATES
# ==========================================

# print("\nNumber of Duplicate Rows:")
# print(df.duplicated().sum())


# ==========================================
# STEP 5: CHECK DATA TYPES
# ==========================================

# print("\nData Types:")
# print(df.dtypes)


# ==========================================
# STEP 6: CATEGORICAL ENCODING
# ==========================================

# Convert Sex into numerical values
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# One-Hot Encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], dtype=int)

# print("\nData After Encoding:")
# print(df.head())


# ==========================================
# STEP 7: FEATURE SELECTION
# ==========================================

# Remove unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# print("\nColumns After Feature Selection:")
# print(df.columns)


# ==========================================
# STEP 8: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# print("\nData After Feature Scaling:")
# print(df.head())



# # ==========================================
# # STEP 9: EXPLORATORY DATA ANALYSIS (EDA)
# # ==========================================

# # Load original dataset for EDA
# eda_df = pd.read_csv(BASE_DIR / "train.csv")


# # ------------------------------------------
# # 9.1: Prepare Data for EDA
# # ------------------------------------------

# # Check the first five rows
# print("\nEDA Dataset:")
# print(eda_df.head())

# # # Check missing values
# print("\nMissing Values Before Data Handling:")
# print(eda_df.isnull().sum())

# # # Fill missing Age values with median
# eda_df['Age'] = eda_df['Age'].fillna(eda_df['Age'].median())

# # # Fill missing Embarked values with mode
# eda_df['Embarked'] = eda_df['Embarked'].fillna(eda_df['Embarked'].mode()[0])

# # # Remove Cabin because it contains many missing values
# eda_df.drop('Cabin', axis=1, inplace=True)

# # # Verify missing values after handling
# print("\nMissing Values After Data Handling:")
# print(eda_df.isnull().sum())


# # ------------------------------------------
# # 9.2: Survival Distribution
# # ------------------------------------------

# # Count the number of passengers who survived and did not survive

# plt.figure(figsize=(6, 4))

# sns.countplot(data=eda_df, x='Survived')

# plt.title("Survival Distribution")
# plt.xlabel("Survived (0 = No, 1 = Yes)")
# plt.ylabel("Number of Passengers")

# plt.show()


# # ------------------------------------------
# # 9.3: Age Distribution
# # ------------------------------------------

# # Show the distribution of passenger ages

# plt.figure(figsize=(8, 5))

# sns.histplot(data=eda_df, x='Age', bins=20, kde=True)

# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Number of Passengers")

# plt.show()


# # ------------------------------------------
# # 9.4: Survival by Gender
# # ------------------------------------------

# # Compare survival between male and female passengers

# plt.figure(figsize=(6, 4))

# sns.countplot(data=eda_df, x='Sex', hue='Survived')

# plt.title("Survival by Gender")
# plt.xlabel("Gender")
# plt.ylabel("Number of Passengers")

# plt.show()


# # ------------------------------------------
# # 9.5: Survival by Passenger Class
# # ------------------------------------------

# # Compare survival across different passenger classes

# plt.figure(figsize=(6, 4))

# sns.countplot(data=eda_df, x='Pclass', hue='Survived')

# plt.title("Survival by Passenger Class")
# plt.xlabel("Passenger Class")
# plt.ylabel("Number of Passengers")

# plt.show()


# # ------------------------------------------
# # 9.6: Correlation Analysis
# # ------------------------------------------

# # Select numerical columns for correlation analysis

# correlation = eda_df.select_dtypes(include='number').corr()

# # Display correlation matrix as a heatmap

# plt.figure(figsize=(10, 7))

# sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')

# plt.title("Correlation Heatmap")

# plt.show()



# =====================================================
# STEP 10: PREPARE DATA FOR MACHINE LEARNING
# =====================================================

X = df.drop('Survived', axis=1)
y = df['Survived']


# =====================================================
# STEP 11: TRAIN-TEST SPLIT
# =====================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# STEP 12: LOGISTIC REGRESSION MODEL
# =====================================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# =====================================================
# STEP 12.1: MAKE PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# STEP 12.2: MODEL EVALUATION
# =====================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:", accuracy)


# =====================================================
# STEP 13: DECISION TREE CLASSIFIER
# =====================================================

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)


# =====================================================
# STEP 13.1: DECISION TREE PREDICTIONS
# =====================================================

y_pred_tree = decision_tree.predict(X_test)

print("\nDecision Tree Predictions:")
print(y_pred_tree)


# =====================================================
# STEP 13.2: DECISION TREE EVALUATION
# =====================================================

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:", tree_accuracy)


# =====================================================
# STEP 14: RANDOM FOREST CLASSIFIER
# =====================================================

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)


# =====================================================
# STEP 14.1: RANDOM FOREST PREDICTIONS
# =====================================================

y_pred_rf = random_forest.predict(X_test)

print("\nRandom Forest Predictions:")
print(y_pred_rf)


# =====================================================
# STEP 14.2: RANDOM FOREST EVALUATION
# =====================================================

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", rf_accuracy)


# =====================================================
# STEP 15: K-NEAREST NEIGHBORS (KNN)
# =====================================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)


# =====================================================
# STEP 15.1: KNN PREDICTIONS
# =====================================================

y_pred_knn = knn.predict(X_test)

print("\nKNN Predictions:")
print(y_pred_knn)


# =====================================================
# STEP 15.2: KNN EVALUATION
# =====================================================

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nKNN Accuracy:", knn_accuracy)


# # =====================================================
# # STEP 16.1: LOGISTIC REGRESSION CLASSIFICATION REPORT
# # =====================================================

from sklearn.metrics import classification_report

# print("\nLogistic Regression Classification Report:")
# print(classification_report(y_test, y_pred))

# # =====================================================
# # STEP 16.2: DECISION TREE CLASSIFICATION REPORT
# # =====================================================

# print("\nDecision Tree Classification Report:")
# print(classification_report(y_test, y_pred_tree))


# =====================================================
# STEP 16.3: RANDOM FOREST CLASSIFICATION REPORT
# =====================================================

# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf))


# =====================================================
# STEP 16.4: KNN CLASSIFICATION REPORT
# =====================================================

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))


# =====================================================
# STEP 17: MODEL COMPARISON
# =====================================================

model_accuracies = {
    "Logistic Regression": accuracy,
    "Decision Tree": tree_accuracy,
    "Random Forest": rf_accuracy,
    "KNN": knn_accuracy
}

print("\nModel Accuracy Comparison:")
for model_name, score in model_accuracies.items():
    print(f"{model_name}: {score:.2%}")