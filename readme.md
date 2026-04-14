
# Week 08 - Monday Assignment  
## Time Series Analysis(e-commerce sales) & Sensor Failure Prediction

---

## 📌 Overview

This assignment covers:
- Time Series Analysis on E-commerce Sales Data  
- Sensor Data Cleaning and Failure Prediction  
- Model Comparison and Business Cost Optimization  

---

# 🟢 EASY SECTION

## 🔹 Sub-step 1: E-commerce Time Series Analysis

### Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("ecommerce_sales_ts.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df = df.sort_values('order_purchase_timestamp')
df.set_index('order_purchase_timestamp', inplace=True)

plt.figure(figsize=(12,5))
plt.plot(df['sales'])
plt.title("Daily Sales Time Series")
plt.show()

result = adfuller(df['sales'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

plot_acf(df['sales'])
plot_pacf(df['sales'])
plt.show()
````

### Result (Example)

* ADF Statistic: -1.23
* p-value: 0.65

### Explanation

* Since p-value > 0.05, the series is **non-stationary**
* The plot shows **trend and possible seasonality**
* ACF/PACF indicate dependence on past values

---

## 🔹 Sub-step 2: Sensor Data Cleaning

### Code

```python
sensor = pd.read_csv("sensor_data.csv")
sensor['timestamp'] = pd.to_datetime(sensor['timestamp'])
sensor = sensor.sort_values('timestamp')

print(sensor.isnull().sum())
print(sensor['timestamp'].duplicated().sum())

sensor = sensor.drop_duplicates(subset=['timestamp'])
sensor.set_index('timestamp', inplace=True)

sensor = sensor.fillna(method='ffill')
sensor = sensor.resample('H').mean()
sensor = sensor.fillna(method='ffill')

sensor.to_csv("sensor_data_cleaned.csv")
```

### Result

* Missing values detected in multiple columns
* Duplicate timestamps found and removed

### Explanation

* Forward fill used to handle missing values
* Resampling ensures consistent time intervals
* Clean data is required for sequence modeling

---

# 🟡 MEDIUM SECTION

## 🔹 Sub-step 3: Baseline Model (ARIMA)

### Code

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

model = ARIMA(train['sales'], order=(1,1,1))
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test['sales'], predictions)
rmse = np.sqrt(mean_squared_error(test['sales'], predictions))

print("MAE:", mae)
print("RMSE:", rmse)
```

### Result (Example)

* MAE: 120.5
* RMSE: 150.3

### Explanation

* ARIMA models time dependency in data
* Differencing handles non-stationarity
* MAE represents average prediction error

---

## 🔹 Sub-step 4: Improved Model (SARIMA)

### Code

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model_sarima = SARIMAX(train['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
model_sarima_fit = model_sarima.fit()

pred_sarima = model_sarima_fit.forecast(steps=len(test))

mae_sarima = mean_absolute_error(test['sales'], pred_sarima)

print("ARIMA MAE:", mae)
print("SARIMA MAE:", mae_sarima)
```

### Result (Example)

* ARIMA MAE: 120.5
* SARIMA MAE: 95.2

### Explanation

* SARIMA captures **weekly seasonality**
* Lower MAE indicates better forecasting
* Improved accuracy helps inventory planning

---

## 🔹 Sub-step 5: Sensor Failure Prediction

### Code

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sensor = pd.read_csv("sensor_data_cleaned.csv")
sensor['timestamp'] = pd.to_datetime(sensor['timestamp'])

sensor['rolling_mean'] = sensor['sensor_1'].rolling(5).mean()
sensor['rolling_std'] = sensor['sensor_1'].rolling(5).std()

sensor['failure_next_24h'] = sensor['machine_status'].shift(-24)
sensor['failure_next_24h'] = sensor['failure_next_24h'].apply(lambda x: 1 if x != 'NORMAL' else 0)

sensor = sensor.dropna()

X = sensor[['rolling_mean', 'rolling_std']]
y = sensor['failure_next_24h']

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

### Result (Example)

* Precision: 0.82
* Recall: 0.90
* F1-score: 0.86

### Explanation

* Rolling features capture sensor trends
* High recall reduces missed failures
* Important due to high cost of failures

---

# 🔴 HARD SECTION

## 🔹 Sub-step 6: Rule vs ML Model

### Code

```python
THRESHOLD = sensor['sensor_1'].mean() + 2 * sensor['sensor_1'].std()
sensor['rule_pred'] = (sensor['sensor_1'] > THRESHOLD).astype(int)

sensor['model_pred'] = model.predict(X)

COST_FN = 1000
COST_FP = 100

def compute_cost(y_true, y_pred):
    cost = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            cost += COST_FN
        elif yt == 0 and yp == 1:
            cost += COST_FP
    return cost

y_true = y

rule_cost = compute_cost(y_true, sensor['rule_pred'])
model_cost = compute_cost(y_true, sensor['model_pred'])

print("Rule Cost:", rule_cost)
print("Model Cost:", model_cost)
```

### Result (Example)

* Rule Cost: 250000
* Model Cost: 180000

### Explanation

* Rule is simple but limited
* ML model captures complex patterns
* Lower cost indicates better decision system

---

## 🔹 Sub-step 7: Cost Optimization

### Code

```python
import numpy as np
from sklearn.metrics import f1_score

NUM_SENSORS = 100000

cost_per_sample = model_cost / len(y_true)
daily_cost = cost_per_sample * NUM_SENSORS

print("Estimated Daily Cost:", daily_cost)

probs = model.predict_proba(X)[:,1]

thresholds = np.linspace(0, 1, 50)

best_cost = float('inf')
best_threshold = 0

for t in thresholds:
    preds = (probs > t).astype(int)
    cost = compute_cost(y_true, preds)
    
    if cost < best_cost:
        best_cost = cost
        best_threshold = t

print("Best Threshold:", best_threshold)

best_f1 = 0
best_f1_threshold = 0

for t in thresholds:
    preds = (probs > t).astype(int)
    f1 = f1_score(y_true, preds)
    
    if f1 > best_f1:
        best_f1 = f1
        best_f1_threshold = t

print("Best F1 Threshold:", best_f1_threshold)
```

### Result (Example)

* Estimated Daily Cost: 1,800,000
* Best Threshold: 0.32
* Best F1 Threshold: 0.55

### Explanation

* Cost-optimal threshold is lower
* F1 treats FP and FN equally
* Business cost prioritizes avoiding failures

---

# ✅ Final Conclusion

* Time series required **proper temporal handling**
* SARIMA improved forecasting by capturing seasonality
* Sensor model focused on **high recall**
* Cost-based optimization is better than F1 for real-world systems


```
