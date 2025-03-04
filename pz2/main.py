import numpy as np
import pandas as pd
import time

# Створимо великий DataFrame
np.random.seed(42)
data_size = 10**6
df = pd.DataFrame({"value": np.random.rand(data_size)})

# Функція для обробки
def transform(x):
    return x ** 2 + np.sin(x) - np.log1p(x)

# Використання .apply()
start_time = time.time()
df["apply_result"] = df["value"].apply(transform)
apply_time = time.time() - start_time

# Використання векторизованої операції NumPy
start_time = time.time()
df["numpy_result"] = transform(df["value"].values)
numpy_time = time.time() - start_time

# Порівняння результатів
print(f"Час виконання через .apply(): {apply_time:.4f} сек")
print(f"Час виконання через NumPy: {numpy_time:.4f} сек")
print(f"Прискорення: {apply_time / numpy_time:.2f}x")
