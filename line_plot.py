import numpy as np
import cv2
import seaborn as sns
import pandas as pd
import csv
import matplotlib.pyplot as plt



data = pd.read_csv('computation_times_frontier_assignment.csv')
#data = pd.read_csv('computation_times_frontier_assignment.csv')

algorithms = set(data['Algorithm'])
grouped = data.groupby(data.Algorithm)
print(grouped.get_group("PRM"))
PRM_data = grouped.get_group("PRM")
Skeleton_data = grouped.get_group("Skeletonize")

ax = sns.barplot(data=data, x='Map', y= 'Computation Time', errorbar='sd', hue= 'Algorithm')
#sns.lineplot(data=Skeleton_data, x='Map', y= 'Computation Time', errorbar='sd', hue= 'Algorithm')
#sns.lineplot(data=Skeleton_data, x='Map', y= 'Computation Time', errorbar='sd')
ax.set(ylabel = 'Seconds')
plt.show()