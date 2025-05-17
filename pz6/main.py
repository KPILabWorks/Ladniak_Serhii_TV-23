import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Raw Data.csv")

df.columns = ['time', 'Bx', 'By', 'Bz', 'B_abs']

df['time'] = df['time'] - df['time'].min()

plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['B_abs'], label='Абсолютне магнітне поле (µT)', color='blue')
plt.title('Зміна абсолютного магнітного поля в часі')
plt.xlabel('Час (с)')
plt.ylabel('Магнітне поле (µT)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

threshold = df['B_abs'].mean() + 2 * df['B_abs'].std()
anomalies = df[df['B_abs'] > threshold]

print("Виявлені аномальні значення магнітного поля:")
print(anomalies[['time', 'B_abs']])
