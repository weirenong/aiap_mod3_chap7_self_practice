# Final Test Score Prediction

## Project Description

This project implements a machine learning pipeline to predict final test scores of students. The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple Regression models are trained and compared to find the best performing one based on various metrics.

---

## Prerequisites and Installation Instructions

### Requirements

* Anaconda or Miniconda
* Python 3.11
* Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/weirenong/aiap_mod3_chap7_self_practice.git
cd aiap_mod3_chap7_self_practice

```


2. Create and activate a conda environment:
```bash
conda create -n aiapenv python=3.11
conda activate aiapenv

```


3. Install the required packages:
```bash
pip install -r requirements.txt

```



---

## Instructions for Executing the Pipeline

To run the complete pipeline, execute:

```bash
python main.py

```

---

## Description of Logical Steps/Flow of the Pipeline

### Logical Flow of the Pipeline

The main script orchestrates the entire pipeline by:

* Loading configuration and data
* Initializing data preparation and model training
* Running the training and evaluation process
* Identifying and evaluating the best model on the test set

### 1. Configuration (`config.yaml`)

The pipeline starts with loading configuration parameters from `src/config.yaml`, which includes:

* Data file path
* Target column name
* Feature categorization (numerical, nominal)
* Train/validation/test split ratios
* Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)

The `DataPreparation` class handles:

* Removing duplicates by student id
* Filling up NaN values for CCA with "None" value
* Filling up NaN values for attendance rate by calculating the average attendance rate for students with similar hours per week values
* Standardizing CCA and Tuition categorical/boolean names
* Dropping rows with NaN final test scores
* Converting sleep time to scalar values in minutes because the original data uses time format and it crosses over midnight causing the data to lose its natural ordering
* Enforces dataframe to only contain columns for training, validation and testing
* Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)

The `ModelTraining` class manages:

* Splitting data into training, validation, and test sets
* Training baseline models (Linear Regression, Ridge, Lasso)
* Hyperparameter tuning for Ridge and Lasso models
* Evaluating models using multiple metrics (MAE, MSE, RMSE, $R^2$)
* Selecting the best model based on $R^2$ score

### 4. Main Execution (`main.py`)

The main script orchestrates the entire pipeline by:

* Loading configuration and data
* Initializing data preparation and model training
* Running the training and evaluation process
* Identifying and evaluating the best model on the test set

---

## Overview of Key Findings from EDA

The exploratory data analysis (EDA) revealed several important insights that guided the feature selection and model development process:

### Target Variable Characteristics
- **Range**: Final test scores span from 32 to 100
- **Central Tendency**: Mean score of 67.2 and median of 68, indicating a roughly normal distribution
- **Variability**: Standard deviation of 14.0 suggests moderate spread in student performance

### Key Predictors of Final Test Scores

**Strong Positive Predictors:**
- **No CCA participation**: Students without co-curricular activities (CCA = None) tend to achieve higher scores
- **Visual learning style**: Students with visual learning preferences outperform auditory learners
- **Tuition attendance**: Students attending tuition show higher median scores
- **High attendance rate (≥91%)**: Strong correlation between attendance and performance
- **Moderate study hours (6-10 hours/week)**: Optimal study time for best performance

**Strong Negative Predictors:**
- **Large class sizes (>20 students)**: Negative correlation with scores
- **Having siblings**: Students with siblings tend to score lower
- **Late sleep time (after 23:30)**: Significantly lower maximum scores
- **Excessive study hours (>15 hours/week)**: May indicate struggling students or burnout

### Features Not Used for Machine Learning Training

The following features were **excluded** from the final model training due to lack of predictive power:

| Feature | Reason for Exclusion |
|---------|----------------------|
| **gender** | Perfectly balanced distribution with identical medians across categories |
| **mode_of_transport** | All transport modes show identical distributions and performance metrics |
| **bag_color** | Uniform distribution across all colors with no correlation to scores |
| **age** | Scores evenly distributed between ages 15-16, no predictive value |
| **wake_time** | No statistical relationship with final test scores |
| **direct_admission** | Excluded to prevent potential data leakage (status may be determined by other features) |

These exclusions were based on statistical analysis showing uniform distributions and identical performance metrics across feature categories, indicating no meaningful relationship with the target variable.

---

## Feature Handling Description

### Data Cleaning

* Missing CCA categories are filled with None as it is assumed that correct categories will be used if it were present. There was existing data with None entered which means the missing data is likely to be None as well.
* Missing attendance rate was filled by taking the average from the students with similar hours per week. It is assumed that attendance for classes contributes to hours per week hence we logically use this as an estimate for the missing values.
* CCA and Tuition had inconsistent values that refer to the same thing so the data was standardised.
* Sleep time was in time format and had to be converted to sleep time minutes because the time crossed over midnight and the data lost its order. Converting it to minutes preserves the order of time.

### Feature Processing

* **Numerical Features:** `n_male`, `n_female`, `sleep_time_mins`, `hours_per_week`, `attendance_rate` used with `StandardScaler` because these features have continuous numerical values that are on different scales.
* **Nominal Features:** `CCA`, `direct_admission`, `learning_style`, `tuition` encoded using `OneHotEncoder` because these features are categorical and have no order.
* **Passthrough Features:** `number_of_siblings` used as-is because it already represents an integer count with a natural, uniform numeric progression.

---

## Explanation of Model Choices

### Models Implemented and Justifications

1. **Linear Regression:** Basic model without regularization to establish a baseline performance and understand the relationship between features and the target variable.
2. **Ridge Regression:** Linear regression with L2 regularization to prevent overfitting, especially useful when dealing with multicollinear features or when we have many features relative to the number of observations.
3. **Lasso Regression:** Linear regression with L1 regularization to perform automatic feature selection by shrinking some coefficients to zero, helping to identify the most important predictors of final test scores.

### Hyperparameter Tuning

* Grid search is performed for Ridge and Lasso models to systematically explore different combinations of hyperparameters and identify the optimal configuration.
* Parameters tuned include `alpha` (regularization strength) and `fit_intercept` (whether to calculate the intercept) to optimize model complexity and fit.
* 5-fold cross-validation is used during tuning to ensure robust model selection by evaluating performance across multiple data splits and reducing the risk of overfitting to a particular train/validation split.

---

## Evaluation of Models

### Evaluation Metrics

* **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual final test scores, providing an interpretable measure of average prediction error in the same units as the target variable.
* **MSE (Mean Squared Error):** Average squared difference between predicted and actual values, which penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.
* **RMSE (Root Mean Squared Error):** Square root of MSE, returning the error metric to the original units of the target variable, making it easier to interpret in practical terms.
* **$R^2$ (Coefficient of Determination):** Proportion of the variance in the final test scores that is predictable from the input features, ranging from 0 to 1 where higher values indicate better model fit and explanatory power.

### Model Performance Comparison

Below is a comparison of the models evaluated in this project, including both baseline models and models with hyperparameter tuning:

#### Baseline Models (Before Hyperparameter Tuning)
| Model              | Alpha | MAE   | MSE   | RMSE  | R²     |
|--------------------|-------|-------|-------|-------|--------|
| Linear Regression  | N/A   | 7.23  | 81.23 | 9.01  | 0.5456 |
| Ridge Regression   | 1     | 7.22  | 81.01 | 9.00  | 0.5459 |
| Lasso Regression   | 1     | 8.06  | 98.12 | 9.90  | 0.4625 |

#### Hyperparameter Tuned Models (After Grid Search)
| Model              | Alpha | MAE   | MSE   | RMSE  | R²     |
|--------------------|-------|-------|-------|-------|--------|
| Ridge Regression   | 1     | 7.18  | 80.12 | 8.95  | 0.5501 |
| Lasso Regression   | 0.01  | 7.20  | 80.50 | 8.97  | 0.5483 |

### Final Model Selection and Interpretation

Based on the evaluation metrics, the best performing model was **Ridge Regression with alpha=1**. This model achieved the lowest MAE, MSE, and RMSE values, and the highest R² score among all evaluated models.

Key findings from the evaluation:
1. **Ridge Regression consistently outperforms Linear and Lasso Regression** in terms of prediction accuracy (lower error metrics) and explanatory power (higher R²).
2. **Hyperparameter tuning improved model performance**, particularly for Lasso Regression which showed a dramatic improvement in all metrics after tuning.
3. **The model explains approximately 55% of the variance** in final test scores, which indicates a moderate level of predictive capability.
4. **Average prediction error** is approximately 7.2 points on the final test score scale, which is a reasonable error for educational prediction tasks.
5. **The best performing model (Ridge with alpha=1)** provides a good balance between low error rates and high explanatory power, making it the most reliable choice for predicting student final test scores.

The model's performance suggests that the selected features (direct admission, CCA, learning style, tuition, number of siblings, class size, sleep time, study hours, and attendance rate) have meaningful relationships with student performance, though there's still substantial variability not captured by the model.

---

## Considerations for Deployment

### Deployment Considerations

#### Model Persistence

To make this project production-ready, implement model persistence:

1. Add model saving functionality to `ModelTraining` class:
```python
import joblib

def save_model(self, model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def load_model(self, filename):
    """Load a trained model from a file."""
    return joblib.load(filename)

```


2. Create a separate inference script (`inference.py`) for making predictions with saved models:
```python
import pandas as pd
import joblib
from src.data_preparation import DataPreparation

# Load configuration and model
# Preprocess new data
# Make predictions

```