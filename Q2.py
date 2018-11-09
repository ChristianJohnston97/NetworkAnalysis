# coding: utf8

from __future__ import division
import random 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import networkx as nx

#-------------------------------------------------
def createVertexGroups(m,k):
	num_nodes = m*k
	vertex_groups = {}
	
	for vertex in range(num_nodes):
		vertex_groups[vertex] = vertex % m

	return num_nodes, vertex_groups

#-------------------------------------------------
def createRingGroupGraph(m, k, p, q):
	num_nodes, groups = createVertexGroups(m,k)
	graph = {}
	for vertex in range(num_nodes):
		#add vertex to graph with empty neighbour set
		graph[vertex] = []

	for vertex1 in range(num_nodes):
		# for all other nodes 
		for vertex2 in range(vertex1 + 1, num_nodes):
			#if in same group or adjacent groups 
			if(groups[vertex1] == groups[vertex2] or abs(groups[vertex1] - groups[vertex2]) == 1 or abs(groups[vertex1] - groups[vertex2]) == m-1):
				# add a new edge with probability p 
				if(random.random() < p):
					graph[vertex1].append(vertex2)
					graph[vertex2].append(vertex1)
			else:
				if(random.random() < q):
					graph[vertex1].append(vertex2)
					graph[vertex2].append(vertex1)

	return graph, num_nodes

#-------------------------------------------------
def make_random_graph(num_nodes, prob):
    #initialize empty graph
    random_graph = {} 
    for vertex in range(num_nodes): random_graph[vertex] = []
    #consider each vertex
    for vertex in range(num_nodes):
        #consider each neighbour with greater value
        for neighbour in range(vertex + 1, num_nodes):
            random_number = random.random()
            #add edge from vertex to neighbour with probability prob
            if random_number < prob:
                #maybe this method for set union is deprecated in python 3
                random_graph[vertex].append(neighbour)        
                random_graph[neighbour].append(vertex)       
    return random_graph

#-------------------------------------
# coauthorship.txt graph
# Num nodes = 1559
# Num edges = 41060
def construct_graph(file):
	startVertices = False
	startEdges = False
	num_edges = 0
	lines = open(file)
	graph = {}
	num_nodes = 0
	for line in lines:
		if startVertices:
			if not startEdges:
				if "*Edges" not in line:
					nodes = line.split(' ')
					node = int(nodes[2])
					graph[node] = []
					num_nodes += 1     
				else:
					startEdges = True                       
			else:
				elements = line.split(' ')
				node1 = int(elements[2])
				node2 = int(elements[3])
				# ignore self loops
				if(node1 == node2):
					continue;
				# ignore duplicate links
				if(node2 not in graph[node1]):
					graph[node1].append(node2)
				if(node1 not in graph[node2]):
					graph[node2].append(node1)
					num_edges += 1
				
		else:
			if "*Vertices" in line:
				startVertices = True
	return graph, num_nodes, num_edges

#--------------------------------------------------------
# Creates a PA graph
def constructPreferentialAttachmentGraph(num_nodes, num_edges_per_node):
    graph = nx.barabasi_albert_graph(num_nodes, num_edges_per_node)
    adj_list = nx.generate_adjlist(graph)

    new_graph = {}
    for vertex in range(num_nodes): 
        new_graph[vertex] = []

    for line in adj_list:
        values = line.split()
        vertex = int(values[0])
        values.pop(0)
        for node in values:
            new_graph[vertex].append(int(node))
            new_graph[int(node)].append(vertex)

    return new_graph, num_nodes

#--------------------------------------
def KBrilliance(graph, node):
	# get all neighbours
	neighbours = graph[node]
	# while its not a star network
	k = len(neighbours)
	while(isStarNetwork(neighbours) == False):
		# calculate degrees of all nodes
		nodeDegrees = calculateNodesDegrees(graph, node, neighbours)
		# get node of greatest degree
		neighbourOfGreatestDegree = max(nodeDegrees, key=nodeDegrees.get)
		# remove it from neighbours
		neighbours.remove(neighbourOfGreatestDegree)
		k = len(neighbours)
		
	return k

#--------------------------------------
# function to check if it is a star network
def isStarNetwork(neighbours):
	for node1 in neighbours:
		for node2 in neighbours:
			if(node1 != node2):
				if(node1 in graph[node2]):
					return False
	return True

#--------------------------------------
def calculateNodesDegrees(graph, node, neighbours):
	#degree of all the neighbouring nodes
	nodeDegrees = {}

	# get all adjacent nodes to the center node
	for node1 in neighbours:
		nodeDegrees[node1] = 0
		for node2 in neighbours:
			if(node1 != node2 and node1 in graph[node2]):
					nodeDegrees[node1] += 1
	return nodeDegrees

#--------------------------------------

def compute_vertex_brilliances(graph):
	brilliance = {}
	#consider each vertex
	for vertex in graph:
		brilliance[vertex] = KBrilliance(graph, vertex)
		print vertex, brilliance[vertex]
	return brilliance

#--------------------------------------

def compute_brilliance_distribution(graph):
	brilliance = compute_vertex_brilliances(graph)
	#initialize dictionary for degree distribution
	brilliance_distribution = {}
	#consider each vertex
	for vertex in brilliance:
		#update brilliance_distribution
		if brilliance[vertex] in brilliance_distribution:
			brilliance_distribution[brilliance[vertex]] += 1
		else:
			brilliance_distribution[brilliance[vertex]] = 1
	return brilliance_distribution
#-------------------------------------------------
# how to compute the normalised value?
def compute_normalised_brilliance_distribution(brilliance_distribution, num_nodes):
	normalised_brilliance_distribution = {}
	for brilliance in brilliance_distribution:
		# normalised_brilliance_distribution
		# divide by the max possible brilliance = num_nodes - 1
		normalised_brilliance_distribution[brilliance] = brilliance_distribution[brilliance] / (num_nodes-1)
	return normalised_brilliance_distribution

#-------------------------------------------------

def plotBrillianceDistribution(normalised_brilliance_distribution):
	#create arrays for plotting
	xdata = []
	ydata = []
	for brilliance in normalised_brilliance_distribution:
		xdata += [brilliance]
		ydata += [normalised_brilliance_distribution[brilliance]]
		
	#plot brilliance distribution 
	plt.xlabel('brilliance')
	plt.ylabel('Normalised Rate')
	plt.title('Vertex Brilliance Distribution of Ring Group Graph')
	plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
	plt.savefig('ring_group_graph_brilliance_distribution.png')

#-------------------------------------------------

# GRAPHS - create graphs with the same number of vertices and edges 
# Num nodes = 1559
# Num edges = 40016 

#graph, num_nodes = createRingGroupGraph(100, 15, 0.8, 0.001)

#graph, num_nodes = constructPreferentialAttachmentGraph(1559, 27)
#print graph
#graph2 = nx.to_networkx_graph(graph)
#print graph2.number_of_nodes()
#print graph2.number_of_edges()

#graph, num_nodes, num_edges = construct_graph("coauthorship.txt")
plt.xlabel('brilliance')
plt.ylabel('Normalised Rate')
plt.title('Vertex Brilliance Distribution of Ring Group Graph')


p = 0.8
q = 0.005
graph, num_nodes = createRingGroupGraph(100, 15, p, q)
brilliance_distribution =  compute_brilliance_distribution(graph)
print "Expected", max(brilliance_distribution, key=brilliance_distribution.get)
normalised_brilliance_distribution = compute_normalised_brilliance_distribution(brilliance_distribution, num_nodes)
#plotBrillianceDistribution(normalised_brilliance_distribution)
xdata = []
ydata = []
for brilliance in normalised_brilliance_distribution:
	xdata += [brilliance]
	ydata += [normalised_brilliance_distribution[brilliance]]
str1 = "p: " + str(p) + " q: " + str(q)
plt.plot(xdata, ydata, marker='.', linestyle='dashed', color='Red', label = str1)

p = 0.6
q = 0.001

graph, num_nodes = createRingGroupGraph(100, 15, p, q)
brilliance_distribution =  compute_brilliance_distribution(graph)
print "Expected", max(brilliance_distribution, key=brilliance_distribution.get)
normalised_brilliance_distribution = compute_normalised_brilliance_distribution(brilliance_distribution, num_nodes)
#plotBrillianceDistribution(normalised_brilliance_distribution)
xdata = []
ydata = []
for brilliance in normalised_brilliance_distribution:
	xdata += [brilliance]
	ydata += [normalised_brilliance_distribution[brilliance]]
str1 = "p: " + str(p) + " q: " + str(q)
plt.plot(xdata, ydata, marker='.', linestyle='dashed', color='Orange', label = str1)


p = 0.4
q = 0.0005

graph, num_nodes = createRingGroupGraph(100, 15, p, q)
brilliance_distribution =  compute_brilliance_distribution(graph)
print "Expected", max(brilliance_distribution, key=brilliance_distribution.get)
normalised_brilliance_distribution = compute_normalised_brilliance_distribution(brilliance_distribution, num_nodes)
#plotBrillianceDistribution(normalised_brilliance_distribution)
xdata = []
ydata = []
for brilliance in normalised_brilliance_distribution:
	xdata += [brilliance]
	ydata += [normalised_brilliance_distribution[brilliance]]
str1 = "p: " + str(p) + " q: " + str(q)
plt.plot(xdata, ydata, marker='.', linestyle='dashed', color='Green', label = str1)

p = 0.2
q = 0.0001

graph, num_nodes = createRingGroupGraph(100, 15, p, q)
brilliance_distribution =  compute_brilliance_distribution(graph)
print "Expected", max(brilliance_distribution, key=brilliance_distribution.get)
normalised_brilliance_distribution = compute_normalised_brilliance_distribution(brilliance_distribution, num_nodes)
#plotBrillianceDistribution(normalised_brilliance_distribution)
xdata = []
ydata = []
for brilliance in normalised_brilliance_distribution:
	xdata += [brilliance]
	ydata += [normalised_brilliance_distribution[brilliance]]
str1 = "p: " + str(p) + " q: " + str(q)
plt.plot(xdata, ydata, marker='.', linestyle='dashed', color='Blue', label = str1)
plt.savefig('ring_group_graph_brilliance_distribution.png')
plt.legend()
plt.show()

