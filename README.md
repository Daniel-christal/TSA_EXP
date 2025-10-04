# Ex.No: 6               HOLT WINTERS METHOD
### Date: 30/9/2025



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
<img width="1657" height="531" alt="image" src="https://github.com/user-attachments/assets/a440b5be-2ca5-4033-ae91-6ad438ef0247" />
<img width="1097" height="524" alt="image" src="https://github.com/user-attachments/assets/df084a7b-d7d2-4887-bcfd-92c83c51bbf5" />
<img width="1191" height="602" alt="image" src="https://github.com/user-attachments/assets/50b5f4f8-c862-4854-be9c-4c6ed7898a3e" />
<img width="984" height="624" alt="image" src="https://github.com/user-attachments/assets/11aeb0b7-f7c1-46ba-a458-81692ffd0abd" />
<img width="1097" height="563" alt="image" src="https://github.com/user-attachments/assets/646fd9d0-9ec9-4eb8-a806-a9b001021d0a" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
