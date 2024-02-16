import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy import stats  # Импорт модуля для расчета корреляции

# Загрузите датасет из CSV-файла
data = pd.read_csv(r'D:\Users\User\Documents\file.csv')

# Извлеките данные для предикторов и зависимой переменной
X = data[['Amy-mPFc', 'Amy-Hipp', 'tetha power']]
y = data['aversion']

# Создайте модель множественной линейной регрессии
model = LinearRegression()

# Параметры для кросс-валидации
num_permutations = 100  # Количество перестановок
kf = KFold(n_splits=5, shuffle=True, random_state=None)

# Пустые списки для хранения результатов
mae_values = []
permuted_mae_values = []
r2_values = []
mse_values = []
correlation_values = []  # Список для хранения коэффициентов корреляции

# Повторите процесс перестановочного теста
for _ in range(num_permutations):
    # Создайте индексы для кросс-валидации, разбивая данные на 5 частей
    mae_iteration = 0
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Рассчитываем MAE на текущей итерации
        mae_iteration += mean_absolute_error(y_test, y_pred)

        # Рассчитываем R-squared и MSE на текущей итерации
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        r2_values.append(r2)
        mse_values.append(mse)

        # Рассчитываем корреляцию и добавляем в список
        correlation = stats.pearsonr(y_test, y_pred)[0]
        correlation_values.append(correlation)

    # Сохраняем средний MAE для этой итерации
    mae_values.append(mae_iteration / 5)  # Делим на 5, так как у нас 5 итераций

    # Теперь выполним перестановку значений aversion
    y_shuffled = np.random.permutation(y)

    # Снова разбиваем данные и обучаем модель на каждой части
    mae_iteration = 0
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Рассчитываем MAE на текущей итерации
        mae_iteration += mean_absolute_error(y_test, y_pred)

    # Сохраняем средний MAE для этой итерации
    permuted_mae_values.append(mae_iteration / 5)  # Делим на 5, так как у нас 5 итераций

# Рассчитайте фактический MAE для исходных данных
kf = KFold(n_splits=5, shuffle=True, random_state=None)
actual_mae = 0
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    actual_mae += mean_absolute_error(y_test, y_pred)

actual_mae /= 5  # Усреднение MAE на 5 итерациях кросс-валидации

# Рассчитайте p-value
p_value = (np.sum(np.array(permuted_mae_values) <= actual_mae) + 1) / (num_permutations + 1)

# Рассчитайте средние значения R-squared (r2) и MSE
mean_r2 = np.mean(r2_values)
mean_mse = np.mean(mse_values)

# Рассчитайте средний коэффициент корреляции
mean_correlation = np.mean(correlation_values)

# Выведите результаты
print("Actual Mean Absolute Error (MAE):", actual_mae)
print("Mean of Permuted MAE Values:", np.mean(permuted_mae_values))
print("p-value:", p_value)
print("Mean R-squared (r2):", mean_r2)
print("Mean Squared Error (MSE):", mean_mse)
print("Mean Correlation Coefficient:", mean_correlation)


