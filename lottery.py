# Import Pandas
import pandas as pd

# Read lottery csv file and store in dataframe
data = pd.read_csv('lottery.csv')

# Store slot names
lottery_slots = ['first','second','third','fourth','fifth','sixth','bonus']

# Initialize frequency Dataframe
frequency_df = pd.DataFrame(columns=['count'],index=range(1,46))

# Replace NaN Values with 0
frequency_df = frequency_df.fillna(0)

# Tabluate count
for slot in lottery_slots:
    slot_data = data[slot].value_counts()
    for key in slot_data.keys():
        frequency_df['count'][key] += slot_data[key]

# Sort numbers in descending count order
frequency_df = frequency_df.sort_values('count',ascending=False)

# TASK-2: Display the most frequently appeared number to the least
for k,v in frequency_df.iterrows():
    n = v['count']
    print(f"{k} -> {n}")

# TASK-3: Create modified lottery data format by adding at new columns.
# Store slot names
lottery_slots = ['first','second','third','fourth','fifth','sixth']
# 1. FEATURE-1 SUM OF DIFFERENCES
# Sum of difference between adjacent numbers
# (e.g [1 6 11 16 21 26] = (6 - 1) + (11 - 6) + (16 - 11) + (21 - 16) + (26 - 21) = 25)
import numpy as np
def sum_of_differences(df):
    x = df[lottery_slots[5]] - df[lottery_slots[4]]
    for i in range(4,0,-1):
        x += df[lottery_slots[i]] - df[lottery_slots[i-1]]
    return x

data['sum_of_differences'] = sum_of_differences(data)

# 2. FEATURE-2 MEAN OF NUMBERS
data['mean'] = data[lottery_slots].apply(np.mean,axis=1)

# 3. FEATURE-3 QUARTER OF YEAR
import datetime
def get_quarter(x):
    ''' Return which quarter of the year the draw was made'''
    x = x.split('.')
    x = list(map(int,x))
    m = x[1]
    if 1 <= m <= 3:
        return 1
    elif 4 <= m <= 6:
        return 2
    elif 7 <= m <= 9:
        return 3
    elif 10 <= m <= 12:
        return 4

data['quarter'] = data['date'].apply(get_quarter)

# Print First 20 Lines
print(data.head(20))

# TASK-4 K-mean clustering – Use any combination of features in your lottery.csv to group all weekly rounds
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# FEAUTURES USED
feature_names = ['sum_of_differences','mean','quarter']
X = data[feature_names]

title = 'lottery_clusters_4'
estimator = KMeans(n_clusters=4)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
estimator.fit(X)
labels = estimator.labels_

ax.scatter(X[feature_names[0]], X[feature_names[1]], X[feature_names[2]],
            c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])
ax.set_title(title)
ax.dist = 12
plt.show()
