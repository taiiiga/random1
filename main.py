from prettytable import PrettyTable
from statistics import mean, variance
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


a0 = 3141592
n = 1000
d = 8
t = 54
p = 91

q = 10**d
k = t + p

a = [a0]
x = []

for i in range(0, n):
    next_value = k * a[i] / q
    a.append(int(str(next_value).split('.')[1]))
    x.append(float('0.' + str(a[i + 1])))

table = PrettyTable(['i', 'ai', 'xi'])
for i in range(0, n):
    table.add_row([i + 1, a[i], x[i]])
print(table)

print('Математическое ожидание:', mean(x))
print('Дисперсия:', variance(x))

interval_size = (max(x) - min(x)) / 10
unique_values = list(set(x))
unique_count_values = len(unique_values)

print('Длина интервала', interval_size)
print('Количество уникальных значений', unique_count_values)
print('Максимум', max(x))
print('Минимум', min(x), '\n')

intervals = []
interval_range = {i: min(x) + interval_size * i for i in range(0, 10)}

for i in range(0, 10):
    print("Интервал #{0}".format(i + 1), end=': ')
    interval = []
    for j in range(0, len(unique_values)):
        if interval_range[i] <= unique_values[j] < interval_range[i] + interval_size:
            interval.append(unique_values[j])
    for value in interval:
        print(value, end=', ')
    intervals.append(interval)
    print()

print()

x = []
y = []
table2 = PrettyTable(['Интервал', 'Количество СВ', 'Частота'])
for i in range(0, 10):
    table2.add_row(["{0} - {1}".format(interval_range[i], interval_range[i] + interval_size), len(intervals[i]), len(intervals[i]) / unique_count_values])
    x.append(interval_range[i])
    y.append(len(intervals[i]) / unique_count_values)
print(table2)

a = np.array(x)
b = np.array(y)
X_Y_Spline = make_interp_spline(a, b)
x_ = np.linspace(a.min(), a.max(), 500)
y_ = X_Y_Spline(x_)

plt.style.use('seaborn')
plt.plot(x_, y_)
plt.ylabel('Частота')
plt.xlabel('Интервал')
plt.show()
