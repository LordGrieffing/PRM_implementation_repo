import numpy as np
import cv2
import seaborn as sns
import pandas as pd
import csv
import matplotlib.pyplot as plt



data = pd.read_csv('computation_times.csv')

grouped = data.groupby(data.Algorithm)
PRM_data = grouped.get_group("PRM")
Skeleton_data = grouped.get_group("Skeletonize")

sns.violinplot(data=Skeleton_data, x='Map', y= 'Computation Time', color='goldenrod')

plt.show()