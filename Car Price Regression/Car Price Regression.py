import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


df=pd.read_csv(BASE_DIR /"CarPrice_Assignment.csv")

# =====================================================
# STEP 1: LOAD THE DATASET
# =====================================================


# print("\nFirst 5 Rows of the Dataset:")
# print(df.head())

# print("\nShape of Dataset:")
# print(df.shape)

# print("\nDataset Information:")
# print(df.info())


# print("No of missing values")
# print(df.isna().sum())

# print("No of duplicates value")
# print(df.duplicated().sum())


# =====================================================
# STEP 2: IDENTIFY TARGET VARIABLE
# =====================================================

target = 'price'

print("\nTarget Variable:", target)

# =====================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# =====================================================

X = df.drop(target, axis=1)
y = df[target]

print("\nFeatures (X):")
print(X.columns)

print("\nTarget (y):")
print(y.name)

# =====================================================
# STEP 4: IDENTIFY CATEGORICAL FEATURES
# =====================================================

categorical_columns = X.select_dtypes(include=['object']).columns

print("\nCategorical Features:")
print(categorical_columns)


# =====================================================
# STEP 5: REMOVE UNNECESSARY FEATURES
# =====================================================

X = X.drop(['car_ID', 'CarName'], axis=1)

print("\nFeatures After Removing Unnecessary Columns:")
print(X.columns)

# =====================================================
# STEP 6: ENCODE CATEGORICAL FEATURES
# =====================================================

X = pd.get_dummies(X, drop_first=True)

print("\nFeatures After Encoding:")
print(X.head())
print("\nNew Shape of X:", X.shape)


# =====================================================
# STEP 7: TRAIN-TEST SPLIT
# =====================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# =====================================================
# STEP 8: LINEAR REGRESSION MODEL
# =====================================================

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

# =====================================================
# STEP 9: MAKE PREDICTIONS
# =====================================================

y_pred = linear_model.predict(X_test)

print("\nPredicted Car Prices:")
print(y_pred[:10])

# =====================================================
# STEP 10: REGRESSION MODEL EVALUATION
# =====================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# =====================================================
# STEP 11: RANDOM FOREST REGRESSION MODEL
# =====================================================

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

# =====================================================
# STEP 12: RANDOM FOREST PREDICTIONS
# =====================================================

y_pred_rf = random_forest.predict(X_test)

print("\nRandom Forest Predicted Prices:")
print(y_pred_rf[:10])

# =====================================================
# STEP 13: RANDOM FOREST MODEL EVALUATION
# =====================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = rf_mse ** 0.5
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression Evaluation:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)


# =====================================================
# STEP 14: MODEL COMPARISON
# =====================================================

print("\nModel Comparison:")
print("-----------------------------------")

print(f"Linear Regression R²: {r2:.4f}")
print(f"Random Forest R²: {rf_r2:.4f}")

print(f"\nLinear Regression MAE: {mae:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")


# =====================================================
# STEP 15: ACTUAL VS PREDICTED PRICES
# =====================================================

comparison = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred_rf
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))

# =====================================================
# STEP 17: FINAL MODEL RESULTS
# =====================================================

print("\n========================================")
print("       CAR PRICE REGRESSION RESULTS")
print("========================================")

print("\nLinear Regression:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("\nRandom Forest Regression:")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")

print("\nBest Model: Random Forest Regressor")