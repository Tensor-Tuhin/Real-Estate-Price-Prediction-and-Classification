# Real-Estate-Price-Prediction-and-Classification
Property Price Prediction using Machine Learning Algorithm and it's deployment using Streamlit

This project predicts:
  - Property Price (Lakhs)
  - Appreciation Rate (r)
  - Investment Category (Good / Bad / Average / Very Good)

It also includes an Analytics Dashboard to explore real-estate trends across India.

Project Structure:
- real_estate.ipynb - This notebook deals with EDA, Data Preprocessing, Feature Engineering, Model Evaluation and its training,
                      and saving the necessary files, encoders and trained models.
- ml_flow.ipynb - This notebook deals with the logging of the models, their metrics (MAE and F1 scores) and parameters into MLFlow.
- re_app.py - This contains the code for the deployment of the streamlit app.

Workflow:
- Cleaning and preparing the dataset.
- Visualisiing the trends using Matplotlib and Seaborn.
- Feature Engineering columns such as r (Appreciation Rate) and investment_type.
- Evaluating which model performs best using cross_val_score.
- Training the best models.
- Saving the necessary files as parquet, encoders and trained models as joblib and feature list as json.
- Logged the models, their metrics and parmaters into MLFlow.
- Built the streamlit app where users can provide input for the desired specifications of their property and thereby know how much and what kind of investments they are going to make.
- An analytics dashboard is also built-in the streamlit app.
