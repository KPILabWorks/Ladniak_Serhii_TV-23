import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Прикладові дані
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'price': [100, 150, 80, 200, 130, 220, 90, 170],
    'discount': [10, 15, 5, 20, 10, 25, 8, 12],
    'units_sold': [500, 400, 600, 300, 450, 250, 550, 420]
})

# Обробка категоріальних даних
data = pd.get_dummies(data, columns=['category'], drop_first=True)

# Розділення на X та y
X = data.drop('units_sold', axis=1)
y = data['units_sold']

# Створення моделі
model = RandomForestRegressor(random_state=42)

# Крос-валідація (5-кратна)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = (-scores)**0.5

# Виведення результатів
print("Оцінка моделі (RMSE на кожній фолді):", rmse_scores)
print("Середнє RMSE:", rmse_scores.mean())
