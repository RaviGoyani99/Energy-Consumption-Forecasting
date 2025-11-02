# Energy Consumption XGBoost Model

## Project Overview
This project aims to predict energy consumption using XGBoost, a popular machine learning algorithm for regression and classification problems. The dataset contains historical energy consumption data, which is used to train the model and make predictions. The main focus is on improving the model's performance by incorporating various features and tuning hyperparameters.

## Technologies
- Python | XGBoost | Scikit-learn | Pandas | NumPy | Matplotlib | Matplotlib | Seaborn | Requests

## Project Description

### Performance Metrics
The following performance metrics were used to assess the model's performance:

- **R-squared (R²)**: Measures the proportion of the variance in the dependent variable that can be explained by the independent variables in the model. R-squared ranges from 0 to 1, where a higher value indicates a better fit of the model to the data. A value of 1 means the model explains all the variability in the data.

- **Root Mean Squared Error (RMSE)**: Measures the average error made by the model in predicting the target variable. It is the square root of the Mean Squared Error and has the same unit as the original values, making it more interpretable than MSE. A lower RMSE value indicates better model performance.

- **Mean Absolute Error (MAE)**: Measures the average absolute difference between the actual values and the predicted values from the model. It calculates how close the predictions are to the actual values by taking the absolute difference (ignoring the sign, positive or negative) and then averaging those differences. MAE is less sensitive to outliers than RMSE. A lower MAE value indicates better model performance.

## Models
Four different models were built and evaluated, each incorporating different features to examine their impact on the model's performance:

1. **Model 1**: Basic XGBoost model using original features from the dataset. This model serves as a baseline for comparison with subsequent models.

2. **Model 2**: Model 1 + holiday data. Holiday data was added to examine whether energy consumption patterns change during holidays, potentially improving the model's predictive accuracy.

3. **Model 3**: Model 2 + lag features. Lag features were introduced to capture the temporal dependencies in the data. These features represent the energy consumption values at previous time steps, which can help the model identify trends and seasonality in the data.

4. **RS Hyperparameter Tuned Model 3**: Model 3 + hyperparameter tuning using Random Search. This model aims to optimize the performance by searching for the best combination of hyperparameters for the XGBoost algorithm.

## Results

| Model | Training RMSE | Training MAE | Training R² | Holdout RMSE | Holdout MAE | Holdout R² | Improvement |
|-------|---------------|--------------|-------------|--------------|-------------|------------|-------------|
| Model 1 | 3,903.05 | 2,930.31 | 0.601 | 3,566.61 | 2,684.74 | 0.698 | Baseline |
| Model 2 | 3,903.05 | 2,930.31 | 0.601 | 3,566.61 | 2,684.74 | 0.698 | 0% |
| Model 3 | 270.26 | 193.81 | 0.998 | 298.83 | 218.36 | 0.998 | 91.6% |
| RS Tuned Model 3 | 306.11 | 223.30 | 0.998 | 335.14 | 247.77 | 0.997 | 90.6% |

*Improvement calculated based on Holdout RMSE reduction from baseline*

## Insights
- **Model 1 and Model 2** have the same performance metrics on both the training and holdout sets, indicating that holiday data did not improve the model's performance. Looking at the feature importance for this model, confirms that the holiday information was not helpful.

- **Model 3**, which included lag features, showed a significant improvement in performance compared to Models 1 and 2. The much higher R-squared (0.998) and lower RMSE and MAE values on both the training and holdout sets indicate that the lag features are valuable predictors and help the model make better predictions by capturing temporal dependencies in the data; looking at the feature importance of the model confirms the observation.

- **RS Hyperparameter Tuned Model 3** demonstrated competitive performance with Model 3. Although it exhibited slightly higher RMSE and MAE values on both training and holdout sets, it maintains an excellent R-squared value (0.997) on the holdout set. The slightly higher error metrics could indicate better generalization and reduced overfitting compared to Model 3.

## Conclusion
Incorporating lag features proved to be the most significant improvement for the model's performance, achieving a 91.6% reduction in RMSE compared to the baseline. The lag features helped capture temporal dependencies in the data, leading to better predictions by identifying trends and seasonality. Hyperparameter tuning using random search provided marginal benefits in terms of model generalization. However, the holiday data did not have a noticeable impact on the model's performance, suggesting that it may not be an important feature for this particular dataset or problem.

## Data Source
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
