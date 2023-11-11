#!/usr/bin/python3

# -- implement RRT algorithm on a image

import cv2
import numpy as np
import networkx as nx
import random as rd
import math
from scipy.optimize import fsolve
import time




g = nx.Graph()

g.add_node(1)
g.add_node(2)
g.add_node(3)



g.add_edge(1, 2)
g.add_edge(2, 3)



edgeList = list(g.edges)
print(g)
print(edgeList)

print(g.nodes)