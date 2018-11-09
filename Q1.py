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
# this uses networkX
def compute_degrees_networkX(graph):
	nodes = graph.nodes()
	return graph.degree(nodes)

#-------------------------------------------------
# this doesnt use networkX
def compute_degrees(graph):
    degree = {}
    for vertex in graph:
        degree[vertex] = 0
    #consider each vertex
    for vertex in graph:
        #amend degree[w] for each outgoing edge from v to w
        for neighbour in graph[vertex]:
            degree[vertex] += 1
    return degree

#-------------------------------------------------
def compute_degree_distribution(graph):
	degree_distribution = {}


	degree = compute_degrees(graph)

	for vertex in degree:
	    if degree[vertex] in degree_distribution:
	        degree_distribution[degree[vertex]] += 1
	    else:
	        degree_distribution[degree[vertex]] = 1
	return degree_distribution

#-------------------------------------------------
def compute_normalised_degree_distribution(degree_distribution, num_nodes):
	normalised_degree_distribution = {}
	for degree in degree_distribution:
		# normalised degree distribution: divide by the max possible degree = num_nodes - 1
		normalised_degree_distribution[degree] = degree_distribution[degree] / (num_nodes-1)
	return normalised_degree_distribution

#-------------------------------------------------
# get range of distribution
def find_range_of_distribution(dd):
	first_element = min(dd, key=int)
	last_element = max(dd, key=int)
	print "Range", last_element - first_element
	print "Expected", max(dd, key=dd.get)
	return first_element, last_element

#-------------------------------------------------
# this gets the normalised degree distribution averaged over multiple trials
def average_normalized_degree_distribution(m, k, p, q, trials):

	# this gets the approximate range (i.e. range for graph)
	graph, num_nodes = createRingGroupGraph(m, k, p, q)
	dd = compute_degree_distribution(graph)
	first_element, last_element = find_range_of_distribution(dd)

	cumulative_dist = {}
	for deg in range(first_element,last_element): cumulative_dist[deg] = 0
	for i in range(trials):
		dd = compute_degree_distribution(graph)
		normalised_degree_distribution = compute_normalised_degree_distribution(dd, num_nodes)
		for deg in range(first_element,last_element):
			if deg in normalised_degree_distribution:
				cumulative_dist[deg] += normalised_degree_distribution[deg]
	average_dist = {}
	for deg in range(first_element,last_element):
		average_dist[deg] = cumulative_dist[deg] / trials
	return average_dist

#------------------------------------------
def plotDegreeDistribution(normalised_degree_distribution, m, k, p, q):

	#create arrays for plotting
	xdata = []
	ydata = []
	for degree in normalised_degree_distribution:
	    xdata += [degree]
	    ydata += [normalised_degree_distribution[degree]]
	    
	#plot degree distribution 
	plt.xlabel('Degree')
	plt.ylabel('Normalised Rate')
	plt.title('Q1 - Degree Distribution of Ring Group Graph')
	colours = ['Red', 'Orange', 'Green', 'Blue', 'Indigo']
	labels = ['p = 0.3, q = 0.2', 'p = 0.35, q = 0.15', 'p = 0.4, q = 0.1', 'p = 0.45, q = 0.05', 'p = 0.5, q = 0']
	p = 0.3
	q = 0.2
	for i in range (5):
		plt.plot(xdata, ydata, marker='.', linestyle='None', color='colours[i], label = labels[i]')
		p += 0.5
		q -= 0.5
	plt.savefig('degree_distribiton_ring_group_graph.png')
	#name = "Q1_m=" + str(m) + "_k="+ str(k) + "_p=" + str(p) + "_q="+ str(q) 
	#plt.savefig(name + ".png")

#------------------------------------------------------
def plotNetwork(G):
	nx.draw(G)
	plt.show()

#------------------------------------------------------
# Function to get average clustering coefficient of the graph
def getClusteringCoefficient(G):

	return nx.average_clustering(G)

#------------------------------------------------------
# this creates a networkX graph from the dictionary
def convertAdjacencyListToNetworkXGraph(graph):

	return nx.to_networkx_graph(graph)

#------------------------------------------------------
# diameter of graph
def computeDiameter(graph):
	return nx.diameter(graph)

#------------------------------------------------------


def run(m, k, p, q):
	#plot degree distribution 
	plt.xlabel('Degree')
	plt.ylabel('Normalised Rate')
	plt.title('Degree Distribution of Ring Group Graph ' + "m: " + str(m) + " k: " + str(k))
	colours = ['Red', 'Orange', 'Green', 'Blue', 'Indigo']
	labels = ['p = 0.3, q = 0.2', 'p = 0.35, q = 0.15', 'p = 0.4, q = 0.1', 'p = 0.45, q = 0.05', 'p = 0.5, q = 0']

	for i in range (5):
		graph, num_nodes = createRingGroupGraph(m, k, p, q)
		print "Number Edges", nx.to_networkx_graph(graph).number_of_edges()

		degree_distribution = compute_degree_distribution(graph)
			
		normalised_degree_distribution = compute_normalised_degree_distribution(degree_distribution, num_nodes)
	

		#create arrays for plotting
		xdata = []
		ydata = []
		for degree in normalised_degree_distribution:
		    xdata += [degree]
		    ydata += [normalised_degree_distribution[degree]]

		plt.plot(xdata, ydata, marker='.', linestyle='None', color= colours[i], label = labels[i])
		p += 0.05
		q -= 0.05

	plt.savefig('degree_distribiton_ring_group_graph.png')
	plt.legend()
	plt.show()


run(40, 100, 0.3, 0.2)






# this is for diameter workings
'''
m =	20
k = 20
q = 0.1
p = 1

sum = 0
for i in range (0,10):
	graph, num_nodes = createRingGroupGraph(m, k, p, q)
	graph = convertAdjacencyListToNetworkXGraph(graph)
	sum += getClusteringCoefficient(graph)
avg = sum /10
print avg
'''

'''
for p in range(1, 11):
	p = p/10
	if p > q:
		graph, num_nodes = createRingGroupGraph(m, k, p, q)
		graph = convertAdjacencyListToNetworkXGraph(graph)
		print p, computeDiameter(graph)
'''



