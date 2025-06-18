# Loan Status Prediction Project

## Overview
This project aims to predict loan outcomes (Paid in Full vs. Charged Off) using historical SBA loan data. By analyzing correlations between numerical features, we identify key factors that influence loan repayment success.

## Problem Statement
The objective is to predict the Loan Status based on historical data by identifying significant correlations between numerical features in the dataset to uncover key factors influencing loan outcomes.

## Dataset
- **Source**: SBAnational.csv
- **Original Size**: Large dataset (reduced to 100,000 rows for manageable training times)
- **Features**: Loan details, borrower information, approval dates, financial amounts, and loan outcomes

## Project Structure

### Data Preprocessing
1. **Data Loading & Exploration**
   - Initial dataset exploration and shape analysis
   - Missing value identification and visualization
   - Duplicate detection

2. **Feature Engineering**
   - Column renaming for better readability
   - Separation of numerical and categorical features
   - Data type conversions and cleaning

3. **Data Cleaning**
   - Removal of constant and non-essential features
   - Handling missing values with mode imputation
   - Date feature processing and extraction
   - Financial amount preprocessing (removing $ and , symbols)

### Feature Categories

#### Numerical Features
- `Loan_Term`: Duration of the loan
- `Number_Employees`: Number of employees in the business
- `Jobs_Created`: Number of jobs created by the loan
- `Jobs_Retained`: Number of jobs retained
- `Disbursed_Amount`: Total loan amount disbursed
- `ChargedOff_Principal`: Amount charged off
- `Approved_Amount`: Gross approved loan amount
- `SBA_Approved_Amount`: SBA guaranteed amount
- `AppYear`, `AppMonth`, `Dismonth`: Extracted date features

#### Categorical Features
- `Borrower_State`: State of the borrower
- `Lender_Name`: Name of the lending institution
- `NAICS_Code`: Industry classification code
- `Business_Status`: Status of the business
- `Urban_Rural_Code`: Location classification
- `Revolving_Credit`: Type of credit facility
- `LowDoc_Program`: Low documentation program participation

## Models Implemented

### 1. Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Split**: 80% training, 20% testing
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: Coefficient analysis

### 2. Neural Network (TensorFlow)
- **Architecture**: 
  - Input layer: Feature dimension
  - Hidden layer 1: 64 neurons (ReLU activation)
  - Hidden layer 2: 32 neurons (ReLU activation)
  - Output layer: 1 neuron (Sigmoid activation)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary crossentropy
- **Training**: 20 epochs, batch size 32

## Key Findings

### Data Insights
- Majority of loans are successfully paid in full
- Strong correlations exist between financial amounts and loan outcomes
- Temporal patterns show variations in approval amounts over years and months

### Model Performance
Both models achieve high accuracy in predicting loan outcomes, with detailed performance metrics available through classification reports and confusion matrices.

### Feature Importance
Key features influencing loan outcomes include:
- `SBA_Approved_Amount`
- `Loan_Term`
- `ChargedOff_Principal`

## Visualizations
The project includes several visualizations:
- Missing value heatmaps
- Correlation matrix heatmaps
- Time series analysis of loan approvals
- Feature importance charts
- Model comparison metrics
- Confusion matrices for both models

## Sensitivity Analysis
Comprehensive sensitivity analysis performed on both models to understand:
- Impact of feature perturbations on model predictions
- Robustness of models to feature changes
- Relative importance of key financial features

## Requirements
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
```

## Usage
1. Ensure the SBAnational.csv file is available in the specified path
2. Run the preprocessing steps to clean and prepare the data
3. Execute either model training section:
   - Logistic Regression for interpretable results
   - Neural Network for potentially higher accuracy
4. Analyze results using the provided visualization and evaluation code

## Model Comparison
The project includes a comprehensive comparison between both models:
- Side-by-side confusion matrices
- Performance metrics comparison
- Feature importance analysis for both approaches

## Future Improvements
- Feature selection optimization
- Hyperparameter tuning
- Cross-validation implementation
- Additional model architectures
- Real-time prediction capabilities

## Notes
- The dataset was reduced to 100,000 rows to manage computational resources
- Financial features required specific preprocessing to handle currency formatting
- Date features were engineered to extract temporal patterns
- Both models show strong predictive capability for loan outcome classification
