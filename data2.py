import gzip
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import wget as wget
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = 'http://93.90.217.253/download/files.synop/27/27612.01.03.2014.12.03.2022.1.0.0.ru.utf8.00000000.xls.gz'
wget.download(url, 'weather.xls')
with gzip.open('weather.xls') as file:
    weather_data_base = pd.read_excel(file, skiprows=6)
# очищаем от пустых строк всю херню в давлении, температуре и влажности
weather_data = weather_data_base[weather_data_base['Po'].notna()]
weather_data = weather_data[weather_data['T'].notna()]
weather_data = weather_data[weather_data['U'].notna()]

weather_data['date'] = pd.to_datetime(weather_data['Местное время в Москве (ВДНХ)'], dayfirst=True)
print(weather_data.head(20))

with plt.ion():
    histogram = weather_data['T'].hist()
    plt.figure(figsize=(10, 5))
    plt.plot(weather_data['date'], weather_data['T'], color='blue', label='Data')
    plt.legend()

    data_short = weather_data[weather_data['date'].between('2016-10-01', '2017-03-01')]
    plt.figure(figsize=(20, 5))
    plt.plot(data_short['date'], data_short['T'], color='red', label='Data')
    plt.legend()
    plt.pause(60)
##############################################First model##############################################
# creating a model
# Новый признак: косинус от дня в году.
# Период [1, 366] перегоним в период [0, 2*pi] (подгоняем косинусоиду по ширине)
# день в году в радианах = (dayofyear - 1) / 366 * 2*pi
# косинус от дня в году = cos(день в году в радианах из диапазона 0 до 2*pi)

# в качестве признака берем день в году
weather_data['dayofyear'] = weather_data['date'].dt.dayofyear
'''
weather_data['cos_dayofyear'] = np.cos((weather_data['dayofyear'] - 1) / 366 * 2 * np.pi)
plt.plot(weather_data['cos_dayofyear'])
plt.show()
data_train = weather_data[weather_data['date'] < '2019-01-01']
data_test = weather_data[weather_data['date'] >= '2019-01-01']
plt.plot(data_train['date'], data_train['T'], color='blue')
plt.plot(data_test['date'], data_test['T'], color='gray')
plt.show()

# обучающая выборка
X_train = pd.DataFrame()
X_train['cos_dayofyear'] = data_train['cos_dayofyear']
# тренировочная выборка
X_test = pd.DataFrame()
X_test['cos_dayofyear'] = data_test['cos_dayofyear']
# "y" оставляем столбцом, как есть
y_train = data_train['T']
y_test = data_test['T']

model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

with plt.ion():
    plt.plot(data_train['date'], data_train['T'], color='blue')
    plt.plot(data_test['date'], data_test['T'], color='gray')
    plt.plot(data_test['date'], pred_test, color='yellow')
    plt.pause(60)
print('Средняя ошибка на тренировочной выборке =', mean_absolute_error(y_train, pred_train))
print('Средняя ошибка на тестовой выборке =', mean_absolute_error(y_test, pred_test))

# Средняя ошибка на тренировочной выборке = 4.302765924386425
# Средняя ошибка на тестовой выборке = 4.393851451336821
'''
##############################################Second model##############################################
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, max_leaf_nodes=9)

# setting limits to training and testing intervals
data_train = weather_data[weather_data['date'] < '2019-01-01']
data_test = weather_data[weather_data['date'] >= '2019-01-01']
# обучающая выборка
X_train_1 = pd.DataFrame()
X_train_1['dayofyear'] = data_train['dayofyear']
X_train_1['T'] = data_train['T']
X_train_1['U'] = data_train['U']
X_train_1['Po'] = data_train['Po']
# тренировочная выборка
X_test_1 = pd.DataFrame()
X_test_1['dayofyear'] = data_test['dayofyear']
X_test_1['T'] = data_test['T']
X_test_1['U'] = data_test['U']
X_test_1['Po'] = data_test['Po']
# "y" оставляем столбцом, как есть
y_train_1 = data_train['T']
y_test_1 = data_test['T']
# обучаем модель
model.fit(X_train_1, y_train_1)
pred_train_1 = model.predict(X_train_1)
pred_test_1 = model.predict(X_test_1)
with plt.ion():
    plt.plot(data_train['date'], data_train['T'], color='blue')
    plt.plot(data_test['date'], data_test['T'], color='gray')
    plt.plot(data_test['date'], pred_test_1, color='yellow')
    plt.pause(60)
print('Средняя ошибка на тренировочной выборке =', mean_absolute_error(y_train_1, pred_train_1))
print('Средняя ошибка на тестовой выборке =', mean_absolute_error(y_test_1, pred_test_1))
plt.figure(figsize=(20, 5))
plt.scatter(data_train['date'], y_train_1, label='Data train')
plt.scatter(data_test['date'], y_test_1, label='Data test')
plt.scatter(data_train['date'], pred_train_1, label='Predict train')
plt.scatter(data_test['date'], pred_test_1, label='Predict test')
plt.show()
# max_depth=5
# Средняя ошибка на тренировочной выборке = 3.8963678284848924
# Средняя ошибка на тестовой выборке = 4.358748955250025

# max_depth=6
# Средняя ошибка на тренировочной выборке = 3.798416785236385
# Средняя ошибка на тестовой выборке = 4.376927266807293

# max_depth=5, min_samples_leaf=3
# Средняя ошибка на тренировочной выборке = 3.8963678284848924
# Средняя ошибка на тестовой выборке = 4.358748955250025

# max_depth=5, min_samples_leaf=3, max_leaf_nodes=9
# Средняя ошибка на тренировочной выборке = 4.028599341775209
# Средняя ошибка на тестовой выборке = 4.262267074344835


# max_depth=5, min_samples_leaf=3, max_leaf_nodes=9 + temperature, pressure, humidity
# Средняя ошибка на тренировочной выборке = 1.3465174626520324
# Средняя ошибка на тестовой выборке = 1.3763548334074398
