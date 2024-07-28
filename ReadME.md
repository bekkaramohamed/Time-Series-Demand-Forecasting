Forecasting Baggage Complaints Using XGBoost and SARIMA Models
Objective 🎯

The aim of this notebook is to forecast baggage complaints over time using two different models: XGBoost and SARIMA. We will compare the results of these two algorithms to determine which one provides better forecasts.
Dataset 📊

The dataset for this analysis is sourced from Kaggle and contains historical data on baggage complaints. This data will be used to train and test the forecasting models.
Approach 🛠️
1. Data Loading and Exploration 🔍

    Load the dataset and explore its structure.
    Perform data preprocessing, including handling missing values and data normalization if necessary.
    Visualize the time series data to understand its characteristics.

2. Modeling with XGBoost 📈

    Prepare the data for XGBoost, which may include creating lag features and train-test splitting.
    Train the XGBoost model on the historical data.
    Evaluate the model performance using appropriate metrics and visualize the results.

3. Modeling with SARIMA 📉

    Perform stationarity tests and differencing if necessary to make the data stationary.
    Identify the best SARIMA parameters (p, d, q) 
    Train the SARIMA model on the historical data.
    Evaluate the model performance using appropriate metrics and visualize the results.



After training and evaluating both models, we observed the following:

    XGBoost Results 📉 
        The XGBoost model struggled to learn the patterns in the time series data.
        The predictions from XGBoost were relatively flat and failed to capture the seasonal and trend components present in the actual data.

    SARIMA Results 📈
        The SARIMA model, on the other hand, performed significantly better.
        It successfully captured both the seasonal and trend components of the time series data.
        The SARIMA model provided accurate forecasts, closely matching the actual values in the test set.

Below is a visual comparison of the actual baggage complaints and the predictions made by both models:

    XGBoost Predictions vs. Actual Values
        The XGBoost predictions are flat and do not reflect the actual variations in the data.
    SARIMA Predictions vs. Actual Values
        The SARIMA predictions closely follow the actual data, showing the model's ability to understand and forecast the underlying patterns.

Conclusion 🏆

In conclusion, while XGBoost is a powerful model for many types of data, it did not perform well in this time series forecasting task. The SARIMA model, with its inherent capability to handle seasonality and trends, outperformed XGBoost by a significant margin. For forecasting baggage complaints, the SARIMA model is the preferred choice. Future work could involve further tuning of the SARIMA parameters or exploring other time series-specific models to improve forecasting accuracy.
