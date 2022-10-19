import random
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

a = 0
b = 4
M = 0.6


def check_func(_x0, _nu):
    if _x0 < 2:
        return True if 0.3 * _x0 > _nu else False
    return True if _nu < 0.2 else False


random_array = list(random.random() for _ in range(10000))
x0_nu = [[a + random.random() * (b - a), random.random() * M] for _ in range(10000)]
verify = []

for x in x0_nu:
    if check_func(x[0], x[1]):
        verify.append(x)

x0_arr = list(map(lambda x: x[0], verify))
nu_arr = list(map(lambda x: x[1], verify))
df = pd.DataFrame({
    'x0': x0_arr,
    'nu': nu_arr
})

print(df)

intervals = np.arange(0.05, 4.01, 0.05)
frequency_table = pd.DataFrame({
    'interval_0': np.array([0.] + list(intervals[:-1])),
    'interval_1': intervals
})

freq = np.zeros(len(intervals))

for x in tqdm(x0_arr):
    for i in range(len(frequency_table)):
        if frequency_table['interval_0'][i] < x <= frequency_table['interval_1'][i]:
            freq[i] += 1
            break

frequency_table['freq'] = freq
frequency_table['relative'] = frequency_table['freq'] / frequency_table['freq'].sum()
print(frequency_table)
sns.barplot(x=frequency_table['interval_0'], y=frequency_table['relative'] * 20)
import matplotlib.pyplot as plt
plt.show()
print(np.array(x0_arr).mean())
print(np.array(x0_arr).var())
print(np.array(nu_arr).mean())
print(np.array(nu_arr).var())
