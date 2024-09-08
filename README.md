# Client Churn Prediction Using PySpark, MLlib, and Azure Databricks

This project builds a churn prediction model using PySpark and MLlib on Azure Databricks. The model predicts whether a client will churn based on usage and feedback data, with the goal of achieving at least 75% accuracy.

## Project Structure

- `notebooks/`: Jupyter notebooks for data preprocessing and model training in Azure Databricks.
- `src/`: Python scripts for data pipeline and model training.
- `data/`: Sample dataset for client data.
- `requirements.txt`: Python dependencies.

## Data Description

The dataset contains the following columns:
- `Tenure`: The number of months the customer has stayed with the company.
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TotalCharges`: The total amount charged to the customer.
- `Gender`: The gender of the customer (Male/Female).
- `Churn`: Whether the customer has churned or not (Yes/No).

## Steps to Run

### 1. Setup Azure Databricks
- Create an Azure Databricks workspace.
- Launch a cluster and import the project notebooks.

### 2. Data Preprocessing
- Use `churn_data_pipeline.ipynb` to preprocess the data by converting categorical variables to numeric and handling missing values.

### 3. Train the Model
- Use `churn_model_training.ipynb` to train the logistic regression model and evaluate its accuracy.

### 4. Save and Deploy the Model
- The trained model is saved in the `/models` directory. You can load and deploy it for real-time predictions.

## Dependencies

Install the following dependencies:
```bash
pyspark==3.1.2
