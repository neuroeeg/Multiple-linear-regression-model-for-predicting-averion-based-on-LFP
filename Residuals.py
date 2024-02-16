# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:48:32 2024

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats

# Загрузите датасет из CSV-файла
data = pd.read_csv(r'D:\Users\User\Documents\file.csv')


# Теперь ваш DataFrame содержит данные с добавленным положительным случайным шумом

# Извлеките данные для предикторов и зависимой переменной
X = data[['Amy-mPFc', 'Amy-Hipp', 'tetha power']]

y = data['aversion']

# Создайте модель множественной линейной регрессии
model = LinearRegression()

# Обучите модель на данных
model.fit(X, y)

# Получите предсказанные значения
predicted_values = model.predict(X)

# Рассчитайте MAE, R-squared и MSE
mae = mean_absolute_error(y, predicted_values)
r2 = r2_score(y, predicted_values)
mse = mean_squared_error(y, predicted_values)

# Рассчитайте коэффициент корреляции Пирсона
correlation_coefficient, _ = stats.pearsonr(y, predicted_values)

# Создайте объект g с помощью sns.jointplot, чтобы настроить параметры осей
g = sns.jointplot(x=y, y=predicted_values, kind="reg", color='#03a5e8', scatter_kws={'s': 120, 'edgecolor': 'none', 'color': (1, 0.4, 0.35, 1)},
                  line_kws={"color": "#03a5e8"}, height=8)

# Установите метки осей и параметры шрифта
g.ax_joint.set_xlabel("Actual Value", fontsize=28)
g.ax_joint.set_ylabel("Predicted Value", fontsize=30)
g.ax_joint.spines['left'].set_linewidth(3)
g.ax_joint.spines['bottom'].set_linewidth(3)
g.ax_joint.spines['right'].set_linewidth(3)
g.ax_joint.spines['top'].set_linewidth(3)
g.ax_joint.tick_params(axis='both', which='major', labelsize=25, width=3, length=6) 

# Выведите MAE, R-squared и корреляцию на график
g.ax_joint.annotate(f'MAE = {mae:.2f}', xy=(0.6, 0.205), xycoords='axes fraction', fontsize=24)
g.ax_joint.annotate(f'R² = {r2:.2f}', xy=(0.6, 0.14), xycoords='axes fraction', fontsize=24)
g.ax_joint.annotate(f'r = {correlation_coefficient:.2f}', xy=(0.6, 0.08), xycoords='axes fraction', fontsize=24)
g.ax_joint.xaxis.labelpad = 20  # Расстояние для "Actual Value"
g.ax_joint.yaxis.labelpad = 20  # Расстояние для "Predicted Value"

# Выведите MAE, R-squared и MAE в консоль
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)
print("Mean Squared Error (MSE):", mse)
print("Correlation Coefficient:", correlation_coefficient)

# Сохраните график
fig = plt.figure(figsize=(8, 6)) 
#plt.savefig("output_plot.png", dpi=300)

# Покажите график
plt.show()