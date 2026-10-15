
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("student-mat.csv", sep=";")
df = df.drop_duplicates()

print("Dataset loaded:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# 3. Identify column types
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()

# 4. Preprocess numerical and categorical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 5. Create the complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-" * 35)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

# 9. Show a few actual and predicted grades
results = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": predictions.round(2)
})

print("\nSAMPLE PREDICTIONS")
print(results.head(10).to_string(index=False))


# 10. Save the trained model
joblib.dump(model, "student_grade_model.pkl")
print("\nTrained model saved as student_grade_model.pkl")

# 11. Export predictions
results.to_csv("prediction_results.csv", index=False)
print("Predictions saved as prediction_results.csv")

# 12. Plot actual vs predicted grades
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Student Grades")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

print("Plot saved as actual_vs_predicted.png")
