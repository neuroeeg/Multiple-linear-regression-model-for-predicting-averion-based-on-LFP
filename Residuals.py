import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Загрузите датасет из CSV-файла
data = pd.read_csv(r'D:\Users\User\Documents\новый_файл.csv')

# Извлеките данные для предикторов и зависимой переменной
X = data[['Amy-mPFc', 'Amy-Hipp', 'tetha power']]
y = data['aversion']

# Создайте модель множественной линейной регрессии
model = LinearRegression()

# Обучите модель на данных
model.fit(X, y)

# Получите предсказанные значения
predicted_values = model.predict(X)

# Рассчитайте MAE
mae = mean_absolute_error(y, predicted_values)

# Получите коэффициенты регрессии
intercept = model.intercept_
coefficients = model.coef_
residuals = y - predicted_values

# Создайте DataFrame для residuals и predicted_values
result_df = pd.DataFrame({'residuals': residuals, 'predicted_values': predicted_values})

# Сохраните DataFrame в файл CSV
#result_df.to_csv('residuals_predicted_values.csv', index=False)

# Создайте график Actual Value vs. Residuals
g = sns.jointplot(data=result_df, x='predicted_values', y='residuals', color=(0.309,0.745, 0.93), height=8)

g.ax_joint.set_xlabel("Predicted Value", fontsize=28)
g.ax_joint.set_ylabel('Residuals', fontsize=30)

# Установите координаты для надписи "Residuals"
g.ax_joint.yaxis.set_label_coords(-0.18, 0.5)
g.ax_joint.xaxis.set_label_coords(0.5, -0.12)
g.plot_joint(sns.scatterplot, color=(1, 0.55, 0.5, 1), s=120, edgecolor='none')
g.ax_joint.axhline(0, color='black', linestyle='--', lw=3)  # Горизонтальная линия на уровне 0 для наглядности
g.ax_joint.spines['left'].set_linewidth(3)
g.ax_joint.spines['bottom'].set_linewidth(3)
g.ax_joint.spines['right'].set_linewidth(3)
g.ax_joint.spines['top'].set_linewidth(3)
g.ax_joint.tick_params(axis='both', which='major', width=3, length=6)
g.ax_joint.tick_params(axis='both', which='both', width=3, length=6)  # Увеличьте размер цифр
g.ax_joint.tick_params(axis='both', which='major', labelsize=25)
fig = plt.figure(figsize=(8, 6)) 
#plt.savefig("output_plot.png", dpi=300)
plt.show()
