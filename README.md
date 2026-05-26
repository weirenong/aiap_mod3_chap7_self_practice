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

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

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




# readme requirements:
Project Title and Description
Begin with the title of your project and a brief description of what it does. This section should provide a high-level overview of your project's purpose and functionality.

Prerequisites and Installation
List any prerequisites needed to run the project, such as Python version, libraries, and other dependencies. Include instructions on how to set up the environment and install the necessary packages. This is where your 'requirements.txt' file will come in handy.

Instructions for Executing the Pipeline
Provide step-by-step instructions for running your end-to-end machine learning pipeline. Include details on how to execute the 'main.py' script and any other relevant scripts. Additionally, explain how to modify any parameters in the 'config.yaml' file if needed.

Description of Logical Steps/Flow of the Pipeline
Describe the logical steps and flow of your machine learning pipeline. Include a brief explanation of each major step, such as data cleaning, preprocessing, model training, evaluation, and prediction. If useful, include flow charts or other visual aids to help illustrate the process.

Overview of Key Findings from EDA
Summarise the key findings from your exploratory data analysis (EDA) conducted in the previous sections. Highlight any important patterns, trends, or insights that influenced the design of your pipeline. Keep the detailed EDA in the Jupyter notebook, and include only a quick summary in the 'README.md'.

Feature Handling Description
Provide a description of how the features in the dataset are cleaned and processed. Include other relevant information such as any newly engineered features and selected features used in model training. Summarise this information in a table for clarity.

Explanation of Model Choices
Explain your choice of models for each machine learning task. Discuss why you selected each model and how it fits the specific requirements of your project.

Evaluation of Models
Evaluate the models you developed, discussing their performance metrics. Explain any metrics used in the evaluation and why they were chosen. This section should provide a clear understanding of how well each model performs.

Considerations for Deployment

things to improve:
Flow Charts / Visual Aids: Your requested structure for the Description of Logical Steps/Flow of the Pipeline mentions: "If useful, include flow charts or other visual aids to help illustrate the process." The current markdown text does not include any image placeholders or diagram links to visually map out the pipeline.

Modifying Configuration Parameters: Your requested structure for the Instructions for Executing the Pipeline section states: "Additionally, explain how to modify any parameters in the 'config.yaml' file if needed." While the config.yaml file is broken down later under the logical flow section, the actual execution instructions only tell the user to run python main.py without explaining how or where to modify parameters before running it.

Feature Handling Table: For the Feature Handling Description, your template explicitly asks to "Summarise this information in a table for clarity." The current extracted content lists these features using standard Markdown bullet points instead of a formatted table.