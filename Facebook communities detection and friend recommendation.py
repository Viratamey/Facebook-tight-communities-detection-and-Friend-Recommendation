# Databricks notebook source
#import data set
fb_edges_file_name = "/FileStore/tables/dataset_facebook_combined.txt"
fb_edges_file = sc.textFile(fb_edges_file_name)
print fb_edges_file.take(88236)


# COMMAND ----------

#retrieve vertex and edges from the data set
def get_user1_tuple(entry):
  row = entry.split(' ')
  return int(row[0])

def get_user2_tuple(entry):
  row = entry.split(' ')
  return int(row[1])

def get_edge_tuple(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1])

# COMMAND ----------

from pyspark.sql import Row
user1RDD = fb_edges_file.map(get_user1_tuple).distinct()
user1Count = user1RDD.count()
print (user1Count)
print 'Vertices: %s' % user1RDD.takeOrdered(5)
user2RDD = fb_edges_file.map(get_user2_tuple).cache().distinct()
user2Count = user1RDD.count()
print (user2Count)
print 'Vertices: %s' % user2RDD.takeOrdered(5)
user1Union2 = user1RDD.union(user2RDD)
userRDD = user1Union2.distinct()
userCount = userRDD.count()
print (userCount)
print 'Vertices: %s' % userRDD.takeOrdered(5)
edgesRDD = fb_edges_file.map(get_edge_tuple).cache().distinct()
ecount = edgesRDD.count()
print (ecount)
print 'edges: %s' % edgesRDD.take(10)


# COMMAND ----------

#import igraph package 
from igraph import *

#build igraph with users and edges from the dataset
vertices = userRDD.collect()
edges = edgesRDD.collect()
g = Graph(vertex_attrs={"label":vertices}, edges=edges, directed=False)


# COMMAND ----------

#overall dataset analysis on the built graph
#check if graph is connected or not and graph is birectional and having all edges have equal weights
print g.is_connected(mode=STRONG)
print g.farthest_points(directed=False, unconn=True, weights=None)
network_diameter = g.diameter(directed=False, unconn=True, weights=None)
print network_diameter
print g.get_diameter(directed=False, unconn=True, weights=None)
network_betweenness = g.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
#reduce method in apache spark to calculate sum of all vertices' betweenness
meannetwork_betweenness= reduce(lambda x, y: x + y, network_betweenness) / len(network_betweenness)
print meannetwork_betweenness

# COMMAND ----------

#check degree distribution of the network
network_degree = g.degree()
print network_degree
#reduce method in apache spark to calculate sum of all vertices' degrees
mean_network_degree= reduce(lambda x, y: x + y, network_degree) / len(network_degree)
print mean_network_degree
from operator import add
network_degreeRDD = sc.parallelize(network_degree)
counts = network_degreeRDD.map(lambda x: (x, 1)).reduceByKey(add)
output = counts.collect()
for (degree, count) in output:
  print("%s %i" % (degree, count))

# COMMAND ----------

#identify insignificant nodes
insignificant_users_list = []
for v in vertices:
  friends_list = g.neighbors(vertex=v, mode=ALL)
  if (len(friends_list) < 2):
    insignificant_users_list.append(v)
print set(insignificant_users_list)

insignificant_users_degree_list=[]
for i in insignificant_users_list:
  insignificant_users_degree_list.append(g.degree(i))
print  set(insignificant_users_degree_list)

#remove island nodes from the graph 
g.delete_vertices(insignificant_users_list)
new_vertices = []
new_edges = []

for v in g.vs:
    new_vertices.append(v["label"])
for e in g.es:
    new_edges.append(e.tuple) 
print len(set(vertices))    
print len(set(insignificant_users_list))    
print len(set(new_vertices))  



# COMMAND ----------

#identify significant nodes
important_users_list = []
important_users_degree_list = []
for v in g.vs:
  v_degree = g.degree(v)
  if(v_degree > 300): 
    important_users_list.append(v.index)
    print v.index
    important_users_degree_list.append(v_degree)
print set(important_users_list)
mean_important_users_degree = reduce(lambda x, y: x + y, important_users_degree_list) / len(important_users_degree_list)
print mean_important_users_degree
#sub graph focussing on node "0" which was identified as significant node
node0_friends_list = g.neighbors(vertex=0, mode=ALL)
freinds_of_friends = g.neighborhood(vertices=0, order=2, mode=ALL)
print len(node0_friends_list)
print len(freinds_of_friends)

node0_friends_list.append(0)
node0_alters = []
user0_graph = g.subgraph(node0_friends_list, implementation = "auto")

for e in user0_graph.es:
    print e.tuple
    node0_alters.append(e.tuple)

# COMMAND ----------

#identify cliques on the subgraph
cliques_user0 = user0_graph.maximal_cliques(min =4 , max =10)
print cliques_user0

# COMMAND ----------

#community detection with centrality based approach using edge betweeness
communities = user0_graph.community_edge_betweenness(directed=False)
clusters = communities.as_clustering()
print communities
print 'a'
print clusters.modularity
print clusters


# COMMAND ----------

#community detection using Newman's leading eigenvector method
clusters = user0_graph.community_leading_eigenvector()
print clusters.modularity
print clusters

# COMMAND ----------

#community detection using the label propagation method of Raghavan et al
clusters = user0_graph.community_label_propagation()
print clusters.modularity
print clusters

# COMMAND ----------

#community detection using the multilevel algorithm of Blondel et al.
multilevelclusters = user0_graph.community_multilevel()
print clusters.modularity
print clusters

# COMMAND ----------

#community detection using the spinglass community detection method of Reichardt & Bornholdt
clusters = user0_graph.community_spinglass()
print clusters.modularity
print clusters

# COMMAND ----------

#community detection using fast greedy algorithm
fastGreedy = user0_graph.community_fastgreedy()
FGcluster = fastGreedy.as_clustering()
print FGcluster.modularity
print FGcluster

# COMMAND ----------

#community detection using walk trap algorithm
walkTrap = user0_graph.community_walktrap() 
WTcluster = walkTrap.as_clustering()
print WTcluster.modularity
print WTcluster

# COMMAND ----------

#community detection using info map algorithm
infoMap = user0_graph.community_infomap()
print infoMap.modularity
print infoMap.as_cover()

# COMMAND ----------

#Part 2 - Friend Recommendation based on clusters detected
#Extract tuples from dataset
def returnTuple(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1]),-1

egoRDD = fb_edges_file.map(returnTuple)
print egoRDD.take(600)

# COMMAND ----------

#Detect no. of mutual friends for any two of the nodes from the graph 
mutual_friends=[]
allusersfriend_list =[]
users = userRDD.collect()
for j in range(len(users)):
    toNodes1  = []
    toNodes2  = []    
    for row in egoRDD.collect():
      if row[0]==users[j]:
        toNodes1.append(row[1])
      elif row[1]==users[j]:
        toNodes2.append(row[0])
    toNodes1RDD = sc.parallelize(toNodes1)
    toNodes2RDD = sc.parallelize(toNodes2)
    user_friendsRDD = toNodes1RDD.union(toNodes2RDD).distinct()
    user_friends = user_friendsRDD.collect()
    allusersfriend_list.append(user_friends)
    
allusersfriend_listRDD = sc.parallelize(allusersfriend_list)  
for i in range(0,len(users)):
  for j in range(i+1,len(users)):
    user1RDD = sc.parallelize(allusersfriend_list[users[i]])
    user2RDD = sc.parallelize(allusersfriend_list[users[j]])
    mutual_between_user1_and_user2 = user1RDD.intersection(user2RDD).distinct()
    count = mutual_between_user1_and_user2.count()
    mutual_friends.append([(users[i], users[j]), count])

mutual_friendsRDD =sc.parallelize(mutual_friends)
sortedRdd = mutual_friendsRDD.sortBy(lambda a: -a[1])
print sortedRdd.collect()
    
    

# COMMAND ----------

allusersfriend_listRDD = allusersfriend_list.collect()  
print allusersfriend_list[users[1]].intersection(allusersfriend_list[users[2]]).distinct()

# COMMAND ----------

#Select one user for whom friend suggestion has to be made
fromuser=115
#Filter mutual friend list for the selected user
suggestions_115_1 = sortedRdd.filter(lambda x:x[0][0]==fromuser).map(lambda x:(x[0][1],x[1]))
print suggestions_115_1.collect()
suggestions_115_2 = sortedRdd.filter(lambda x:x[0][1]==fromuser).map(lambda x:(x[0][0],x[1]))
print suggestions_115_2.collect()
suggestions_115 = suggestions_115_1.union(suggestions_115_2)
print suggestions_115.collect()
suggestions_115_sorted = suggestions_115.sortBy(lambda x:-x[1])
print suggestions_115_sorted.collect()
suggestions_115_RDD = suggestions_115_sorted.map(lambda x:x[0])
print suggestions_115_RDD.collect()

# COMMAND ----------

#Get all friends of user 115
friends_115_1= egoRDD.filter(lambda x:x[0]==fromuser).map(lambda x:x[1])
friends_115_2= egoRDD.filter(lambda x:x[1]==fromuser).map(lambda x:x[0])
friends_115 = friends_115_1.union(friends_115_2)
print friends_115.collect()
#Get all non friends of user 115
already_friends = suggestions_115_RDD.intersection(friends_115)
suggestions = suggestions_115_RDD.subtract(already_friends)
print suggestions.collect()

# COMMAND ----------

#Narrow down suggestion based on communities
#Communities detected by fastgreedy is opted because of better modularity
suggestion_list = suggestions.collect()
community_based_suggestion=[]
for cluster_index in range(8):
  for member in suggestion_list:
    if member in multilevelclusters[cluster_index] and 115 in multilevelclusters[cluster_index]:
      community_based_suggestion.append(member)

print community_based_suggestion
