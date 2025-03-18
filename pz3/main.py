import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Генерація синтетичного набіру даних
np.random.seed(42)
timestamps = np.arange(0, 1000, 1)
values = np.sin(0.01 * timestamps) + np.random.normal(scale=0.1, size=len(timestamps))
data = pd.DataFrame({'timestamp': timestamps, 'value': values})


# Методи децимації
def downsample_mean(data, factor):
    return data.groupby(data.index // factor).mean()


def downsample_median(data, factor):
    return data.groupby(data.index // factor).median()


def downsample_decimate(data, factor):
    return pd.DataFrame({'value': decimate(data['value'], factor)})


# Функція прогнозування
def predict_and_evaluate(train_x, train_y, test_x, test_y):
    model = LinearRegression()
    model.fit(train_x.reshape(-1, 1), train_y)
    pred_y = model.predict(test_x.reshape(-1, 1))
    return mean_squared_error(test_y, pred_y), pred_y


# Коефіцієнт децимації
factor = 10

# Децимація даних
methods = {
    'Mean': downsample_mean,
    'Median': downsample_median,
    'Decimate': downsample_decimate
}

results = {}

for method_name, method in methods.items():
    downsampled_data = method(data, factor).reset_index(drop=True)
    timestamps_decimated = timestamps[::factor][:len(downsampled_data)]  # Коригування міток

    train_size = int(len(downsampled_data) * 0.8)
    train_x, train_y = timestamps_decimated[:train_size], downsampled_data['value'][:train_size]
    test_x, test_y = timestamps_decimated[train_size:], downsampled_data['value'][train_size:]

    mse, pred_y = predict_and_evaluate(train_x, train_y, test_x, test_y)
    results[method_name] = (mse, pred_y, test_x, test_y)

# Побудова графіка
plt.figure(figsize=(12, 6))
for method_name, (mse, pred_y, test_x, test_y) in results.items():
    plt.plot(test_x, test_y, label=f'{method_name} Real')
    plt.plot(test_x, pred_y, '--', label=f'{method_name} Predicted')
    print(f'MSE for {method_name}: {mse:.4f}')

plt.xlabel('Часова мітка')
plt.ylabel('Значення')
plt.legend()
plt.title('Графік децимації даних')
plt.show()
