import numpy as np
import cv2
#import seaborn
#import pandas as pd
import csv

algorithms = ['Skeletonize', 'PRM']
map_names = ['Map1', 'Map2', 'Map3', 'Map4', 'Map5']


PRM_Data = []
Skeleton_Data = []

for i in range(5):
    PRM_File = open(file = 'PRM_profile_data/PRM_profile_sequence' + str(i+1) + '_n100.txt', mode='r')
    Skeleton_File = open(file = 'Skeleton_profile_data/Skeleton_profile_sequence' + str(i+1) + '_n100.txt', mode='r')

    PRM_Data.append(PRM_File.readlines())
    Skeleton_Data.append(Skeleton_File.readlines())



computation_times = {
    'Skeletonize': {
        'Map1' : Skeleton_Data[0],
        'Map2' : Skeleton_Data[1],
        'Map3' : Skeleton_Data[2],
        'Map4' : Skeleton_Data[3],
        'Map5' : Skeleton_Data[4]
    },

    'PRM': {
        'Map1' : PRM_Data[0],
        'Map2' : PRM_Data[1],
        'Map3' : PRM_Data[2],
        'Map4' : PRM_Data[3],
        'Map5' : PRM_Data[4]
    }
}

csv_filename = 'computation_times.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['Algorithm', 'Map', 'Run', 'Computation Time']
    csv_writer.writerow(header)

    # Write data rows
    for algorithm in algorithms:
        for map_name in map_names:
            times = computation_times[algorithm][map_name]
            for run, time in enumerate(times, start=1):
                row = [algorithm, map_name, run, time]
                csv_writer.writerow(row)

print(f'Data has been written to {csv_filename}')




