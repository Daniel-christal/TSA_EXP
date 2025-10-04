# Ex.No: 6               HOLT WINTERS METHOD
### Date: 04.10.2025



### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv('Clean_Dataset.csv', index_col=0)


data_monthly = data['price']   


data_monthly.plot()
plt.show()


scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)

scaled_data.plot()
plt.show()


decomposition = seasonal_decompose(scaled_data, model="additive", period=12)
decomposition.plot()
plt.show()

scaled_data = scaled_data + 1 
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]



model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()


test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Test RMSE:", rmse)

print("Scaled data std (sqrt):", np.sqrt(scaled_data.var()))
print("Scaled data mean:", scaled_data.mean())

final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(scaled_data) / 4)) 

ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly observations')
ax.set_ylabel('Values')
ax.set_title('Prediction')
plt.show()

```
### OUTPUT:
<img width="847" height="586" alt="image" src="https://github.com/user-attachments/assets/77a8d351-7679-4a44-8228-c6e05c1c7377" />

<img width="672" height="492" alt="image" src="https://github.com/user-attachments/assets/739b6e75-568c-4039-bd93-f36a6ed1c296" />

<img width="702" height="530" alt="image" src="https://github.com/user-attachments/assets/6eb7cdf0-7016-44b2-931f-a7e61a2da5e7" />

<img width="761" height="650" alt="image" src="https://github.com/user-attachments/assets/b2c71290-5b47-4850-8c1b-8f2c1973fa11" />

<img width="761" height="555" alt="image" src="https://github.com/user-attachments/assets/1a919483-59c6-4fc7-adbd-293726d49024" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
