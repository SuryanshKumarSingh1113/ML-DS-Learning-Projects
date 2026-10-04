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
