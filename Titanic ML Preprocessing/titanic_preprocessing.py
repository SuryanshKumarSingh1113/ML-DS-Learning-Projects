import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# STEP 1: LOAD AND UNDERSTAND THE DATA
# ==========================================

df = pd.read_csv("train.csv")

print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# STEP 2: CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


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
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==========================================
# STEP 4: CHECK DUPLICATES
# ==========================================

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# STEP 5: CHECK DATA TYPES
# ==========================================

print("\nData Types:")
print(df.dtypes)


# ==========================================
# STEP 6: CATEGORICAL ENCODING
# ==========================================

# Convert Sex into numerical values
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# One-Hot Encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], dtype=int)

print("\nData After Encoding:")
print(df.head())


# ==========================================
# STEP 7: FEATURE SELECTION
# ==========================================

# Remove unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print("\nColumns After Feature Selection:")
print(df.columns)


# ==========================================
# STEP 8: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nData After Feature Scaling:")
print(df.head())



# ==========================================
# STEP 9: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Load original dataset for EDA
eda_df = pd.read_csv("train.csv")


# ------------------------------------------
# 9.1: Prepare Data for EDA
# ------------------------------------------

# Check the first five rows
print("\nEDA Dataset:")
print(eda_df.head())

# Check missing values
print("\nMissing Values Before Data Handling:")
print(eda_df.isnull().sum())

# Fill missing Age values with median
eda_df['Age'] = eda_df['Age'].fillna(eda_df['Age'].median())

# Fill missing Embarked values with mode
eda_df['Embarked'] = eda_df['Embarked'].fillna(eda_df['Embarked'].mode()[0])

# Remove Cabin because it contains many missing values
eda_df.drop('Cabin', axis=1, inplace=True)

# Verify missing values after handling
print("\nMissing Values After Data Handling:")
print(eda_df.isnull().sum())


# ------------------------------------------
# 9.2: Survival Distribution
# ------------------------------------------

# Count the number of passengers who survived
# and did not survive

plt.figure(figsize=(6, 4))

sns.countplot(data=eda_df, x='Survived')

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------
# 9.3: Age Distribution
# ------------------------------------------

# Show the distribution of passenger ages

plt.figure(figsize=(8, 5))

sns.histplot(data=eda_df, x='Age', bins=20, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------
# 9.4: Survival by Gender
# ------------------------------------------

# Compare survival between male and female passengers

plt.figure(figsize=(6, 4))

sns.countplot(data=eda_df, x='Sex', hue='Survived')

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------
# 9.5: Survival by Passenger Class
# ------------------------------------------

# Compare survival across different passenger classes

plt.figure(figsize=(6, 4))

sns.countplot(data=eda_df, x='Pclass', hue='Survived')

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------
# 9.6: Correlation Analysis
# ------------------------------------------

# Select numerical columns for correlation analysis

correlation = eda_df.select_dtypes(include='number').corr()

# Display correlation matrix as a heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Correlation Heatmap")

plt.show()


