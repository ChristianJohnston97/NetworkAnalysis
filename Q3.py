import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import networkx as nx

# information at each vertex: vertex id and the number of neighbours 
# Choose parameters so that the graphs are very likely to be connected

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

#--------------------------------------------------------
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

#--------------------------------------------------------
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
                # add a new edge with probability q 
                if(random.random() < q):
                    graph[vertex1].append(vertex2)
                    graph[vertex2].append(vertex1)

    return graph, groups, num_nodes
# function to query node

#--------------------------------------------------------
# this creates a networkX graph from the dictionary
def convertAdjacencyListToNetworkXGraph(graph):

    return nx.to_networkx_graph(graph)

#--------------------------------------------------------
def queryNode(graph, node):
    return len(graph[node])

#--------------------------------------------------------
def ringGroupGraphTraversal(graph, vertex_groups, m, k, p, q, startVertex, targetVertex):
    
    # vertex_groups is a dictionary 
    # key is node, value is what group it is in
    totalNumQueries = 0
    currentVertex = startVertex

    group_of_target_vertex = vertex_groups[targetVertex]

    if(p == q):
        while currentVertex != targetVertex:
            random_node = random.choice(graph[currentVertex])
            totalNumQueries += 1
            # move to selected node
            currentVertex = random_node
        return totalNumQueries

    if(p > q):
        while currentVertex != targetVertex:

            # Randomise
            random.shuffle(graph[currentVertex])

            for vertex in graph[currentVertex]:

                # if neighbour is target vertex, return 
                if(vertex == targetVertex):
                    return totalNumQueries

                # get group of vertex
                group = vertex_groups[vertex]

                # is this one query??
                totalNumQueries += 1

                # want to move to group of target vertex or adjacent group
                if(group == group_of_target_vertex or (abs(group_of_target_vertex-group) == 1) or (abs(group_of_target_vertex - group) == m-1)):
                    currentVertex = vertex
                    break;
                else:
                    continue;

            else:
                random_node = random.choice(graph[currentVertex])
                currentVertex = random_node

        return totalNumQueries

    if(q > p):
        while currentVertex != targetVertex:

            # Randomise
            random.shuffle(graph[currentVertex])

            for vertex in graph[currentVertex]:

                # if neighbour is target vertex, return 
                if(vertex == targetVertex):
                    return totalNumQueries

                # get group of vertex
                group = vertex_groups[vertex]

                # is this one query??
                totalNumQueries += 1

                # want to move to non-same and non-adjacent group
                if(group != group_of_target_vertex and abs(group_of_target_vertex-group) != 1 and abs(group_of_target_vertex - group != m-1)):
                    currentVertex = vertex
                    break;
                else:
                    continue;

            else:
                random_node = random.choice(graph[currentVertex])
                currentVertex = random_node  

        return totalNumQueries     
       
    return totalNumQueries

#--------------------------------------------------------
# Random Graphs Only
def randomTraversal(graph, startVertex, targetVertex):
    totalNumQueries = 0
    currentVertex = startVertex
    while currentVertex != targetVertex:
        random_node = random.choice(graph[currentVertex])
        totalNumQueries += 1
        # move to selected node
        currentVertex = random_node

    return totalNumQueries
#--------------------------------------------------------

# Random Graphs Only
def randomTraversal2(graph, startVertex, targetVertex):
    totalNumQueries = 0
    currentVertex = startVertex
    found = False;
    while found == False:

        for vertex in graph[currentVertex]:
            totalNumQueries += 1

            if(vertex == targetVertex):
                found = True
                break;
        else:
            random_node = random.choice(graph[currentVertex])
            currentVertex = random_node     

    return totalNumQueries

#--------------------------------------------------------
# Preferential Attatchment Traversal
def paTraversal(graph, m, startVertex, targetVertex):

    totalNumQueries = 0
    currentVertex = startVertex
  
    while currentVertex != targetVertex:
        # past queries at a given node
        pastQueries = {}

        # Randomise
        random.shuffle(graph[currentVertex])

        for vertex in graph[currentVertex]:

            # query a node: number of neighbours and vertex group
            numNeighbours = queryNode(graph, vertex)
            totalNumQueries += 1

            # if neighbour is target vertex, return 
            if(vertex == targetVertex):
                return totalNumQueries

              
            # Save the query: number of neighbours and its group
            pastQueries[vertex] = numNeighbours

        # this is to generate random with a bias proportional to degree
        flattenedList = []

        for key in pastQueries:
            for i in range(pastQueries[key]):
                flattenedList.append(key)
        

        # prefer nodes with the greatest number of neighbours- biased random
        node = random.choice(flattenedList)
        currentVertex = node 

    return totalNumQueries

#--------------------------------------------------------
def runSearch(graph, startVertex, targetVertex ):
    # checks to see if graph is connected
    networkXgraph = convertAdjacencyListToNetworkXGraph(graph)
    if not nx.is_connected(networkXgraph):
        return False

    print "Start Vertex: ", startVertex
    print "Target Vertex: ", targetVertex

    #numQueries = ringGroupGraphTraversal(graph, groups, m, k, p, q, startVertex, targetVertex)
    numQueries = paTraversal(graph, m, startVertex, targetVertex)
    #numQueries = randomTraversal2(graph, startVertex, targetVertex)

    print "Number of queries: ", numQueries

    return numQueries

#--------------------------------------------------------
def runSearchOverForEveryNode(graph):
    cummulativeScore = {}
    networkXgraph = convertAdjacencyListToNetworkXGraph(graph)
    nodes = networkXgraph.nodes()
    totalTime = 0
    counter = 0
    for node1 in nodes:
        for node2 in nodes:
            counter +=1 
            if(node1 != node2):
                num_queries = runSearch(graph, int(node1), int(node2))
                totalTime += num_queries
                if num_queries in cummulativeScore:
                    cummulativeScore[num_queries] += 1
                else:
                    cummulativeScore[num_queries] = 1
    print totalTime/counter
    return cummulativeScore
            
#--------------------------------------------------------

def plotSearchTime(cummulativeScore):

    #create arrays for plotting
    xdata = []
    ydata = []
    for time in cummulativeScore:
        xdata += [time]
        ydata += [cummulativeScore[time]]
        
    #plot degree distribution 
    plt.xlabel('Search Time')
    plt.ylabel('Number of instances')
    plt.title('Search Time For Preferential Attatchment Graph')
    #plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig("searchTime.png")


#--------------------------------------------------------
# PA Graph Stuff
num_nodes = 100
m = 50
graph, num_nodes = constructPreferentialAttachmentGraph(num_nodes, m)


# Random Graph Stuff 
#num_nodes = 100
#p = 0.9
#graph = make_random_graph(num_nodes, p)

# Ring Group Graph Stuff 
#m = 20
#k = 5
#p = 0.4
#q = 0.1
#graph, groups, num_nodes = createRingGroupGraph(m, k, p, q)  

cummulativeScore = runSearchOverForEveryNode(graph)
plotSearchTime(cummulativeScore)

