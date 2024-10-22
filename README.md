# Hospital Readmission Prediction for Diabetes Patients

## Overview

This repository contains the Python code for predicting hospital readmission within 30 days for patients diagnosed with diabetes, based on data from 130 US hospitals covering ten years (1999-2008). The project is divided into two main parts: building a basic predictive model and developing an improved model using classification and clustering techniques.

## Objective

The goal is to accurately predict early hospital readmissions (<30 days) of diabetes patients using machine learning models. The project aims to showcase data preprocessing, visualization, model building, evaluation, and enhancement through clustering.

## Data

The dataset, `diabetic_data.csv`, consists of 47 features and 101766 instances representing hospital records, laboratory medications, and patient stays up to 14 days.
## File Descriptions

- `basicModel.py`: Contains the implementation of the basic predictive model. This script demonstrates data cleaning, transformation, feature selection, model training (using a linear model), and evaluation with a focus on handling missing values aggressively and dealing with imbalanced data through a sampling technique.

- `improvedModel.py`: Implements the improved model with a comprehensive approach towards data cleansing, feature engineering, and model tuning. It introduces cluster-based classification using the K-Means algorithm to enhance model performance and provides a detailed comparison between the basic and improved models.

## Usage

### Data Preprocessing

1. Aggressive cleaning: Removing columns with high missing values, replacing '?' with NaN, and dropping near-zero variance columns.
2. Feature engineering: Converting the 'readmitted' column into a binary feature for prediction.
3. Data balancing: Techniques like oversampling to balance the dataset.

### Model Building

- Basic Model: A linear model such as Logistic Regression is used focusing on selected predictors, evaluated using cross-validation and performance metrics suitable for imbalanced datasets.
- Improved Model: Explores advanced preprocessing, utilizes the entire dataset, and implements cluster-based classification for enhanced performance.

### Evaluation

Models are evaluated based on accuracy, precision, recall, F1-score, and the area under the ROC curve. The improved model's performance is also compared against the basic model to highlight the effectiveness of the clustering approach.


## How to Run

Ensure you have Python 3 and all necessary libraries installed. Run the following commands in your terminal:

```bash
pip install -r requirements.txt
python basicModel.py
python improvedModel.py
