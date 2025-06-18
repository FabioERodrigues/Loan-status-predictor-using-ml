Loan Status Prediction Project
Overview
This project implements machine learning models to predict loan status outcomes (Paid in Full vs. Charged Off) based on historical Small Business Administration (SBA) loan data. The analysis focuses on identifying key factors that influence loan performance through correlation analysis and predictive modeling.
Problem Statement
The objective is to predict the Loan Status (e.g., Paid in Full or Charged Off) based on historical data. By identifying significant correlations between numerical features in the dataset, we aim to uncover key factors influencing loan outcomes.
Dataset

Source: SBAnational.csv
Original Size: Large dataset (reduced to 100,000 rows for computational efficiency)
Target Variable: Loan Status (Binary: Paid in Full / Charged Off)
Features: Various loan, borrower, and lender characteristics

Data Preprocessing
Feature Engineering

Renamed columns for better readability and understanding
Created new temporal features:

AppYear: Application year
AppMonth: Application month
Dismonth: Disbursement month



Data Cleaning

Removed constant and irrelevant features (Loan_ID, Franchise_Status, Borrower_Name, Default_Date)
Handled missing values using mode imputation for categorical features
Converted financial features from string format (with $ and commas) to float
Encoded categorical variables using integer mapping
Removed outliers and invalid dates

Feature Categories

Numerical Features: Loan amounts, terms, employee counts, dates
Categorical Features: Borrower location, lender information, program types
Financial Features: Disbursed amount, outstanding amount, SBA approved amount

Models Implemented
1. Logistic Regression

Algorithm: Scikit-learn LogisticRegression
Preprocessing: StandardScaler normalization
Train/Test Split: 80/20

2. Neural Network (TensorFlow)

Architecture:

Input layer: Feature dimension
Hidden layer 1: 64 neurons (ReLU activation)
Hidden layer 2: 32 neurons (ReLU activation)
Output layer: 1 neuron (Sigmoid activation)


Optimizer: Adam (learning rate: 0.001)
Loss Function: Binary crossentropy
Training: 20 epochs, batch size 32

Key Findings
Feature Importance Analysis
Through sensitivity analysis, the most important features for loan status prediction are:

SBA_Approved_Amount: Amount guaranteed by SBA
Loan_Term: Duration of the loan
ChargedOff_Principal: Principal amount charged off

Model Performance
Both models demonstrate strong predictive performance with detailed confusion matrices and classification reports provided for evaluation.
Visualizations
The project includes several data visualizations:

Correlation Heatmap: Shows relationships between numerical features
Time Series Analysis: Application year/month vs. SBA approved amounts
Feature Importance Charts: Visual representation of feature significance
Confusion Matrices: Model performance evaluation

Requirements
python# Core libraries
pandas
numpy
matplotlib
seaborn

# Machine Learning
scikit-learn
tensorflow

# Data Processing
csv
datetime
Usage

Data Loading: Place SBAnational.csv in the appropriate directory
Run Preprocessing: Execute data cleaning and feature engineering sections
Model Training: Train both Logistic Regression and Neural Network models
Evaluation: Analyze model performance using provided metrics
Sensitivity Analysis: Examine feature importance and model sensitivity

File Structure
├── realfinalcoursework.py     # Main analysis script
├── SBAnational.csv           # Dataset (not included)
└── README.md                 # This file
Model Evaluation Metrics

Accuracy: Overall prediction accuracy
Precision: Positive prediction accuracy
Recall: True positive rate
F1-Score: Harmonic mean of precision and recall
Confusion Matrix: Detailed prediction breakdown

Sensitivity Analysis
The project includes comprehensive sensitivity analysis to understand:

How changes in key features affect model predictions
Feature importance through permutation testing
Impact of feature value modifications on accuracy

Future Improvements

Implement additional machine learning algorithms (Random Forest, Gradient Boosting)
Perform hyperparameter tuning
Add cross-validation for more robust evaluation
Include feature selection techniques
Expand dataset size for better generalization

Notes

The dataset is limited to 100,000 rows to manage computational complexity
Missing values are handled through mode imputation for categorical features
Financial features are converted from string format to numerical values
The analysis focuses on binary classification (Paid in Full vs. Charged Off)

License
This project is for educational and research purposes.
