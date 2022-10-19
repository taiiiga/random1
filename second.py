from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import random


x = [2, 3, 5, 12, 21, 33, 44]
p = [0.1, 0.15, 0.2, 0.05, 0.02, 0.33, 0.15]
n = 1000

intervals_1 = [round(sum(p[:i + 1]), 2) for i in range(len(p))]
intervals_0 = [0] + intervals_1[:-1]
rnd_numbers = [random.random() for _ in range(n)]
freq = []

for x in rnd_numbers:
    for i in range(0, n):
        if intervals_0[i] < x <= intervals_1[i]:
            freq.append(1)
            break

print(intervals_0)

mean = 0
sum1 = 0

for i in range(0, len(x)):
    mean = mean + x[i] * p[i]
    sum1 = sum1 + x[i] * x[i] * p[i]

variance = sum1 - mean * mean

print('Математическое ожидание:', mean)
print('Дисперсия:', variance)

interval_size = (max(x) - min(x)) / 7
unique_values = list(set(x))
unique_count_values = len(unique_values)
intervals = []
interval_range = {i: min(x) + interval_size * i for i in range(0, 7)}

for i in range(0, 7):
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
for i in range(0, 7):
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
