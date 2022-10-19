import random
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

xi = [5, 8, 13, 16, 21, 24, 29]
pi = [0.1, 0.02, 0.25, 0.15, 0.35, 0.03, 0.1]

intervals_1 = [round(sum(pi[:i + 1]), 2) for i in range(len(pi))]
intervals_0 = [0] + intervals_1[:-1]

frequency_table = pd.DataFrame({
    'p': pi,
    'x': xi,
    'interval_0': intervals_0,
    'interval_1': intervals_1,
})

rnd_numbers = [random.random() for _ in range(10000)]

freq = np.zeros(len(frequency_table))

for x in rnd_numbers:
    for i in range(len(frequency_table)):
        if frequency_table['interval_0'][i] < x <= frequency_table['interval_1'][i]:
            freq[i] += 1
            break

frequency_table['frequency'] = freq.astype('int')
frequency_table['relative'] = frequency_table['frequency'] / 10000

print(frequency_table)

mat = 0
for i in range(len(frequency_table)):
    mat += frequency_table['relative'][i] * frequency_table['x'][i]

print(mat)

mat_sq = 0
for i in range(len(frequency_table)):
    mat_sq += frequency_table['relative'][i] * (frequency_table['x'][i]) ** 2

var = mat_sq - mat ** 2

print(var)

sns.barplot(x=frequency_table['x'], y=frequency_table['frequency']);

import matplotlib.pyplot as plt
plt.show()