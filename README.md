
# SWYNEX End-to-End Machine Learning Project

## Project Overview
This project predicts students' final mathematics grades using machine learning. It uses the Student Performance dataset and a Random Forest Regressor.

## Features
- Dataset loading and inspection
- Data preprocessing for numerical and categorical features
- Training and evaluation of a Random Forest regression model
- Evaluation using MAE, RMSE, and R-squared
- Saving the trained model
- Exporting predictions to CSV
- Visualizing actual versus predicted grades
- Loading the saved model for prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Project Files
- `main.py` — trains and evaluates the model
- `predict.py` — loads the saved model and generates a sample prediction
- `student-mat.csv` — Student Performance mathematics dataset
- `student_grade_model.pkl` — saved trained model
- `prediction_results.csv` — actual and predicted grades
- `actual_vs_predicted.png` — prediction visualization
- `requirements.txt` — required Python libraries

## Setup Instructions

1. Install Python.
2. Clone or download this repository.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the training script:

   ```bash
   python main.py
   ```

5. Run the prediction script:

   ```bash
   python predict.py
   ```

## Model Evaluation
The current test run produced these results:

- MAE: 1.19
- RMSE: 2.01
- R-squared: 0.804


## Note
The model uses earlier grades (`G1` and `G2`) along with other student information to predict the final mathematics grade (`G3`). It is an educational demonstration and should not be used to make high-stakes decisions about students.

## Author

**Shreya Kumari**

Machine Learning Project — SWYNEX Technologies

