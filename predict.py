
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_grade_model.pkl")

# Load a sample student record from the dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Use the first student's information, excluding final grade G3
student = df.drop(columns=["G3"]).iloc[[0]]

# Predict the final maths grade
predicted_grade = model.predict(student)[0]

print("STUDENT GRADE PREDICTION")
print("-" * 30)
print("Predicted final maths grade:", round(predicted_grade, 2))
