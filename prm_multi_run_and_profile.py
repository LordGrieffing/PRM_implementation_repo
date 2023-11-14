#!/usr/bin/python3

# -- implement RRT algorithm on a image

import cv2
import numpy as np
import networkx as nx
import random as rd
import math
from scipy.optimize import fsolve
import time



def generateSample(height, width):
    sample_x = rd.randrange(0, height)
    sample_y = rd.randrange(0, width)

    #print([sample_x, sample_y])
    
    return [sample_x, sample_y]

# -- gets the largest node from a graph
def getLargestNode(Exgraph):
    list_of_Nodes = list(Exgraph.nodes)
    largestNode = 0

    # Check if the graph is empty
    if not list_of_Nodes:
        return largestNode
    
    for i in range(len(list_of_Nodes)):
        if list_of_Nodes[i] >= largestNode:
            largestNode = list_of_Nodes[i]

    return largestNode

# -- This method uses Bresenham's line Algorithm to draw an edge
# -- I got this from ChatGPT
def bresenham_line(x1, y1, x2, y2):
    # Setup initial conditions
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Decision variables for determining the next point
    if dx > dy:
        err = dx / 2
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x, y))  # Add the final point
    return points

# -- Checks if a node is a neighbor of another node
def check_if_neighbor(currentNode, neighborNode, radius):

    nodeDist = math.dist(currentNode, neighborNode)

    if nodeDist <= radius:
        return True
    
    else:
        return False

# -- Checks if a legal edge can be built between two nodes    
def check_legal_edge(currentNode, neighborNode, img):
    line = []
    line = bresenham_line(int(currentNode[0]), int(currentNode[1]), int(neighborNode[0]), int(neighborNode[1]))
    lineValid = True

    for j in range(len(line)):
        if np.all(img[line[j][0], line[j][1]] == 0):
            lineValid = False
            break

    return lineValid


# -- Checks if legal edges can be added to a node with any of the other nodes in the graph
def add_legal_edges(Exgraph, newNode, radius, img):

    nodeList = list(Exgraph)
    newNodeCoords = [0, 0]
    currentNodeCoords = [0, 0]
    newNodeCoords[0] = Exgraph.nodes[newNode]['x']
    newNodeCoords[1] = Exgraph.nodes[newNode]['y']

    if len(nodeList) == 1:
        return 

    for i in range(len(nodeList)):
        currentNodeCoords[0] = Exgraph.nodes[nodeList[i]]['x']
        currentNodeCoords[1] = Exgraph.nodes[nodeList[i]]['y']

        neighbor = check_if_neighbor(newNodeCoords, currentNodeCoords, radius)

        if neighbor:
            legal_edge = check_legal_edge(newNodeCoords, currentNodeCoords, img)

            if legal_edge:
                Exgraph.add_edge(newNode, nodeList[i])

    return 






# -- Does the PRM algorithm
def prm(Exgraph, imgHeight, imgWidth, img, sampleNum = 100, radius = 40):
    
    graph_unfinished = True

    largestNode = getLargestNode(Exgraph)
    newMax = largestNode + sampleNum

    # -- Run a while loop until the graph has the number of nodes specified by sampleNum
    while graph_unfinished:

        # -- Generate a sample
        legal_sample = False
        newSample = [0,0]

        while not legal_sample:
            
            newSample = generateSample(imgHeight, imgWidth)
            color = img[int(newSample[0]), int(newSample[1])]

            if not np.all(color == 0):
                legal_sample = True
        
        # -- Add legal node to graph
        largestNode = largestNode + 1
        Exgraph.add_node(largestNode)
        Exgraph.nodes[largestNode]['x'] = newSample[0]
        Exgraph.nodes[largestNode]['y'] = newSample[1]
        #print(Exgraph)

        # -- add legal edges to sample
        add_legal_edges(Exgraph, largestNode, radius, img)

        # -- check if graph is finished
        #print(Exgraph)
        if len(Exgraph.nodes) == newMax:
            graph_unfinished = False

    return Exgraph

def erode_image(map_array, filter_size = 9):
    """ Erodes the image to reduce the chances of robot colliding with the wall
    each pixel is 0.05 meter. The robot is 30 cm wide, that is 0.3 m. Half is
    0.15 m. If we increase the walls by 20 cm on either side, the kernel should
    be 40 cm wide. 0.4 / 0.05 = 8
    """
    kernel = np.ones((filter_size,filter_size), np.uint8)
    eroded_img = cv2.erode(map_array, kernel, iterations = 1)
    return eroded_img

def profileEnd(start, path):
    end = time.time()
    total = end - start
    with open(path, 'a') as f:
        f.write('\n')
        f.write(str(total))
        
       
























if __name__ == "__main__":
    # -- import an image and convert it to a binary image
    #img = cv2.imread('maze5.png')

    img = []

    for i in range(5):
        img.append(cv2.imread('map_sequences/map_sequence_' + str(i + 1) + '.png'))
    
    # -- build empty graph
    Exgraph = nx.Graph()

    # In this situation all the maps are the same resolution so it doesn't matter which you grab the resolution from
    imgHeight, imgWidth, channels = img[0].shape

    # -- Loop over each sample 100 times sequentially
    # -- Like 1, 2, 3, 4 ,5 then again 1, 2, 3, 4, 5 repeat 100 times
    for i in range(100):
        
        # -- Reset the graph

        Exgraph = nx.Graph()

        for j in range(5):
            # -- map 1
            start_timer = time.time()
            map_data = erode_image(img[j])
            Exgraph = prm(Exgraph, imgHeight, imgWidth, map_data, 40, 20)
            profileEnd(start_timer, 'PRM_profile_data/PRM_profile_sequence' + str(j + 1) + '_n100.txt')
            print(Exgraph)


    '''
    # -- Display graph
    nodeList = list(Exgraph.nodes)
    edgeList = list(Exgraph.edges)

    # -- add nodes
    for i in range(len(nodeList)):
        nodeCoords = (Exgraph.nodes[nodeList[i]]['y'], Exgraph.nodes[nodeList[i]]['x'])
        cv2.circle(img, nodeCoords, 4, (255, 0, 0), -1)

    for i in range(len(edgeList)):
        
        currentEdge = edgeList[i]
        cv2.line(img, (Exgraph.nodes[currentEdge[0]]['y'], Exgraph.nodes[currentEdge[0]]['x']), (Exgraph.nodes[currentEdge[1]]['y'], Exgraph.nodes[currentEdge[1]]['x']), (0, 0, 255), 1)


    # -- Display image
    cv2.imshow('My Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    







































