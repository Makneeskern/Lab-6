# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:04:36 2019

@author: Mikef
"""

from scipy.interpolate import interp1d
import sys


class DisjointSetForest:
    def __init__(self, n):
        self.forest = [-1] * n

    def is_index_valid(self, index):
        return 0 <= index < len(self.forest)

    # Problem 6
    def find(self, a):
        if not self.is_index_valid(a):
            return -1

        if self.forest[a] < 0:
            return a

        return self.find(self.forest[a])

    # Problem 7
    def find_contains_loop(self, a, s=None):
        if not self.is_index_valid(a):
            return -1

        if s is None:
            s = set()

        if a in s:
            print("Loop found")
            return -1

        s.add(a)

        if self.forest[a] < 0:
            return a

        return self.find_contains_loop(self.forest[a], s)

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:  # Problems 3 and 4
            self.forest[rb] = ra

    # Problem 2
    def in_same_set(self, a, b):
        if not self.is_index_valid(a) or not self.is_index_valid(b):
            return False

        return self.find(a) == self.find(b)

    # Problem 5
    def num_sets(self):
        count = 0

        for k in self.forest:
            if k < 0:
                count += 1

        return count

    # Problem 9
    def is_equal(self, dsf):

        if len(self.forest) != len(dsf.forest):
            return False

        # Running time O(n^2). Can you do it in O(n)? (:
        for i in range(len(self.forest)):
            for j in range(len(self.forest)):
                if self.in_same_set(i, j) != dsf.in_same_set(i, j):
                    return False

        return True

    def __str__(self):
        return str(self.forest)
    

class GraphAM:

    def __init__(self, vertices, weighted=False, directed=False):
        self.am = []

        for i in range(vertices):  # Assumption / Design Decision: 0 represents non-existing edge
            self.am.append([0] * vertices)

        self.directed = directed
        self.weighted = weighted
        self.representation = 'AM'

    def is_valid_vertex(self, u):
        return 0 <= u < len(self.am)

    def insert_vertex(self):
        for lst in self.am:
            lst.append(0)

        new_row = [0] * (len(self.am) + 1)  # Assumption / Design Decision: 0 represents non-existing edge
        self.am.append(new_row)

        return len(self.am) - 1  # Return new vertex id

    def insert_edge(self, src, dest, weight=1):
        if not self.is_valid_vertex(src) or not self.is_valid_vertex(dest):
            return

        self.am[src][dest] = weight

        if not self.directed:
            self.am[dest][src] = weight

    def delete_edge(self, src, dest):
        self.insert_edge(src, dest, 0)

    def num_vertices(self):
        return len(self.am)

    def display(self):
        print('[', end='')
        for i in range(len(self.am)):
            print('[', end='')
            for j in range(len(self.am[i])):
                edge = self.am[i][j]
                if edge != 0:
                    print('(' + str(j) + ',' + str(edge) + ')', end='')
            print(']', end=' ')
        print(']')

    def draw(self):
        scale = 30
        fig, ax = plt.subplots()
        for i in range(len(self.am)):
            for j in range(len(self.am[i])):
                edge = self.am[i][j]

                if edge != 0:
                    d, w = j, edge
                    if self.directed or d > i:
                        x = np.linspace(i * scale, d * scale)
                        x0 = np.linspace(i * scale, d * scale, num=5)
                        diff = np.abs(d - i)
                        if diff == 1:
                            y0 = [0, 0, 0, 0, 0]
                        else:
                            y0 = [0, -6 * diff, -8 * diff, -6 * diff, 0]
                        f = interp1d(x0, y0, kind='cubic')
                        y = f(x)
                        s = np.sign(i - d)
                        ax.plot(x, s * y, linewidth=1, color='k')
                        if self.directed:
                            xd = [x0[2] + 2 * s, x0[2], x0[2] + 2 * s]
                            yd = [y0[2] - 1, y0[2], y0[2] + 1]
                            yd = [y * s for y in yd]
                            ax.plot(xd, yd, linewidth=1, color='k')
                        if self.weighted:
                            xd = [x0[2] + 2 * s, x0[2], x0[2] + 2 * s]
                            yd = [y0[2] - 1, y0[2], y0[2] + 1]
                            yd = [y * s for y in yd]
                            ax.text(xd[2] - s * 2, yd[2] + 3 * s, str(w), size=12, ha="center", va="center")
            ax.plot([i * scale, i * scale], [0, 0], linewidth=1, color='k')
            ax.text(i * scale, 0, str(i), size=20, ha="center", va="center",
                    bbox=dict(facecolor='w', boxstyle="circle"))
        ax.axis('off')
        ax.set_aspect(1.0)
        plt.show()


import matplotlib.pyplot as plt
import numpy as np

# import graph_AM as graph # Replace line 3 by this one to demonstrate adjacy maxtrix implementation
# import graph_EL as graph # Replace line 3 by this one to demonstrate edge list implementation

class Edge:
    def __init__(self, dest, weight=1, source = ""):
        self.dest = dest
        self.weight = weight
        self.src = source


class GraphAL:
    # Constructor
    def __init__(self, vertices, weighted=False, directed=False):
        self.al = [[] for i in range(vertices)]
        self.weighted = weighted
        self.directed = directed
        self.representation = 'AL'

    def is_valid_vertex(self, u):
        return 0 <= u < len(self.al)

    def insert_vertex(self):
        self.al.append([])

        return len(self.al) - 1  # Return id of new vertex

    def insert_edge(self, source, dest, weight=1):
        if not self.is_valid_vertex(source) or not self.is_valid_vertex(dest):
            print('Error, vertex number out of range')
        elif weight != 1 and not self.weighted:
            print('Error, inserting weighted edge to unweighted graph')
        else:
            self.al[source].append(Edge(dest, weight, source))
            if not self.directed:
                self.al[dest].append(Edge(source, weight, dest))

    def delete_edge(self, source, dest):
        if source >= len(self.al) or dest >= len(self.al) or source < 0 or dest < 0:
            print('Error, vertex number out of range')
        else:
            deleted = self._delete_edge(source, dest)
            if not self.directed:
                deleted = self._delete_edge(dest, source)
            if not deleted:
                print('Error, edge to delete not found')

    def _delete_edge(self, source, dest):
        i = 0
        for edge in self.al[source]:
            if edge.dest == dest:
                self.al[source].pop(i)
                return True
            i += 1
        return False

    def num_vertices(self):
        return len(self.al)

    def display(self):
        print('[', end='')
        for i in range(len(self.al)):
            print('[', end='')
            for edge in self.al[i]:
                print('(' + str(edge.dest) + ',' + str(edge.weight) + ')', end='')
            print(']', end=' ')
        print(']')

    def draw(self):
        scale = 30
        fig, ax = plt.subplots()
        for i in range(len(self.al)):
            for edge in self.al[i]:
                d, w = edge.dest, edge.weight
                if self.directed or d > i:
                    x = np.linspace(i * scale, d * scale)
                    x0 = np.linspace(i * scale, d * scale, num=5)
                    diff = np.abs(d - i)
                    if diff == 1:
                        y0 = [0, 0, 0, 0, 0]
                    else:
                        y0 = [0, -6 * diff, -8 * diff, -6 * diff, 0]
                    f = interp1d(x0, y0, kind='cubic')
                    y = f(x)
                    s = np.sign(i - d)
                    ax.plot(x, s * y, linewidth=1, color='k')
                    if self.directed:
                        xd = [x0[2] + 2 * s, x0[2], x0[2] + 2 * s]
                        yd = [y0[2] - 1, y0[2], y0[2] + 1]
                        yd = [y * s for y in yd]
                        ax.plot(xd, yd, linewidth=1, color='k')
                    if self.weighted:
                        xd = [x0[2] + 2 * s, x0[2], x0[2] + 2 * s]
                        yd = [y0[2] - 1, y0[2], y0[2] + 1]
                        yd = [y * s for y in yd]
                        ax.text(xd[2] - s * 2, yd[2] + 3 * s, str(w), size=12, ha="center", va="center")
            ax.plot([i * scale, i * scale], [0, 0], linewidth=1, color='k')
            ax.text(i * scale, 0, str(i), size=20, ha="center", va="center",
                    bbox=dict(facecolor='w', boxstyle="circle"))
        ax.axis('off')
        ax.set_aspect(1.0)
        plt.show()


# import graph_AM as graph # Replace line 3 by this one to demonstrate adjacy maxtrix implementation
# import graph_EL as graph # Replace line 3 by this one to demonstrate edge list implementation











def kruscals_algo(graph):
    if str(type(graph)) == "<class '__main__.GraphAM'>":
        return kaam(graph)
    if str(type(graph)) == "<class '__main__.GraphAL'>":
        return kaal(graph)
    print("The \"graph\" privided was neither an AL or AM graph.")
    return None

def kaam(graph):
    edge_list = []
    for shelf in range(len(graph.am)):
        for item in range(len(graph.am[shelf])):
            if graph.am[shelf][item] != 0:
                edge_list.append(Edge(item, graph.am[shelf][item], shelf))
    sorted_list = sort_edges_h(edge_list)
    solution = GraphAM(len(graph.am), graph.weighted, graph.directed)
    visited = DisjointSetForest(len(graph.am))
    for edge in sorted_list:
        if not visited.in_same_set(edge.src, edge.dest):
            solution.insert_edge(edge.src, edge.dest, edge.weight)
            visited.union(edge.src,edge.dest)
    return solution

def kaal(graph):
    sorted_list = sort_edges(graph)
    solution = GraphAL(len(graph.al), graph.weighted, graph.directed)
    visited = DisjointSetForest(len(graph.al))
    for edge in sorted_list:
        if not visited.in_same_set(edge.src, edge.dest):
            solution.insert_edge(edge.src, edge.dest, edge.weight)
            visited.union(edge.src,edge.dest)
    return solution

def sort_edges(graph):
    ordered_edges = []
    for i in range(len(graph.al)):
        ordered_edges +=  graph.al[i]
    if graph.weighted:
        ordered_edges = sort_edges_h(ordered_edges)
    return ordered_edges

def sort_edges_h(edge_list):
    #Despite "Heap Sort" being the fastest sort I know of(Taking roughly O(n)), I decided to just use Merge sort because it's easier to read,
    #won't use ass much memory (by my estimation), and is still reasonably fast.
    
    if len(edge_list) <= 1:
        return edge_list
    
    half_edge_l = edge_list[0 : (len(edge_list)//2)]
    half_edge_r = edge_list[len(edge_list)//2 : len(edge_list)]
    
    half_edge_l = sort_edges_h(half_edge_l)
    half_edge_r = sort_edges_h(half_edge_r)
    
    left_index = 0
    right_index = 0
    new_list = []
    while left_index < len(half_edge_l) and right_index < len(half_edge_r):
        if half_edge_l[left_index].weight < half_edge_r[right_index].weight:
            new_list.append(half_edge_l[left_index])
            left_index += 1
        else:
            new_list.append(half_edge_r[right_index])
            right_index += 1
    if left_index >= len(half_edge_l):
        new_list = new_list + half_edge_r[right_index : len(half_edge_r)]
    elif right_index >= len(half_edge_r):
        new_list = new_list + half_edge_l[left_index : len(half_edge_l)]
    else:
        print("an Error occured while sorting. Program cannot be safely continue")
        sys.exit(-1)
    return new_list



def topo_sort(graph):
    if str(type(graph)) == "<class '__main__.GraphAM'>":
        return tsam(graph)
    if str(type(graph)) == "<class '__main__.GraphAL'>":
        return tsal(graph)
    print("The \"graph\" privided was neither an AL or AM graph.")
    return None

def tsam(graph):
    in_degree = [0] * len(graph.am)
    for item in range(len(graph.am)):
        for shelf in range(len(graph.am)):
            if graph.am[shelf][item] != 0:
                in_degree[item] += 1
    
    visited_order = []
    qeue = []
    zero = False
    while True:
        for vertex in range(len(in_degree)):
            if in_degree[vertex] == 0:
                in_degree[vertex] = -1
                qeue.append(vertex)
                zero = True
        if not zero:
            print("Topological sort not technically possible")
            return -1
        if qeue == []:
            break
        visited_order.append(qeue[0])
        
        for item in range(len(graph.am[visited_order[-1]])):
            if graph.am[visited_order[-1]][item] != 0:
                in_degree[item] = in_degree[item] - 1
        qeue.remove(qeue[0])
    return visited_order
    
def tsal(graph):
    in_degree = [0] * len(graph.al)
    for vertex in graph.al:
        for edge in vertex:
            in_degree[edge.dest] += 1
    
    visited_order = []
    qeue = []
    zero = False
    while True:
        for vertex in range(len(in_degree)):
            if in_degree[vertex] == 0:
                in_degree[vertex] = -1
                qeue.append(vertex)
                zero = True
        if not zero:
            print("Topological sort not technically possible")
            return -1
        if qeue == []:
            break
        visited_order.append(qeue[0])
        
        for edge in graph.al[visited_order[-1]]:
            in_degree[edge.dest] = in_degree[edge.dest] - 1
        qeue.remove(qeue[0])
    return visited_order


if __name__ == "__main__":
    plt.close("all")
    g = GraphAL(6, True, True)
    g.insert_edge(0, 1, 12)
    g.insert_edge(0, 3, 20)
    g.insert_edge(0, 5, 3)
    g.insert_edge(1, 2, 5)
    g.insert_edge(1, 3, 7)
    g.insert_edge(2, 0, 10)
    g.insert_edge(2, 3, 17)
    g.insert_edge(2, 4, 22)
    g.insert_edge(3, 5, 13)
    g.insert_edge(4, 3, -1)
    #for shelf in g.am:
    #    for item in shelf:
    #        print(item, end = ", ")
    #    print()
    #print()
    #for shelf in kruscals_algo(g).am:
    #    for item in shelf:
    #        print(item, end = ", ")
    #    print()
    #print()
    print(topo_sort(g))
    

