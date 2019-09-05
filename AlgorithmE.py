from RangeTree import *


def algorithmE(G, separatorList, edges, X, Y):

    graph = Graph(edges)
    path = []
    path_X = []
    distances = []
    distances_X =[]
    rangeTree = RangeTree()
    # TODO: [E1] Base case: if n\ln(n) < 4k(k+1) then find all distances using Dijkstra's Algorithm and terminate

    # [E2] Distances from separator: Compute d(z,v) for every z that belongs to the separator,
    #      using k applications of Dijkstra's algorithm

    # In distances is stored the distances (d(z,v)) from every
    # separator vertex z to every other vertex in V(G)
    for i in separatorList:
        p1, d1 = graph.dijkstra(i, 1)
        path.append(p1)
        distances.append(d1)

    # [E3] Add shortcuts
    # TODO: maybe skip

    # [E4.1] Start iterating over {z1,...,zk}
    # [E4.2] Build range tree for zi. Construct a k-dimensional
    # range tree R of points {p(y) = (p1,...,pk)}
    # The output is stored at pi.
    j = 0
    pj = []
    pi = []
    temp = []
    for i in range(0, len(separatorList)): # build range tree for zi
        for y in Y:
            for j in range(0,len(separatorList)):
                temp.append(distances[i][y] - distances[j][y])
            pj.append(temp)
            temp = []
        pi.append(pj)
        pj = []

    # TODO: f(p(y)) = d(zi, y) using the monoid (Z,max)
    rangeTree.initialization2(pi)

    # [E4.3] Query range tree.
    # For each x in X, query R with l1 = ... = lk = inf
    # Output: e(x, zi; Y)

    # Calculating the distances from x to every other node
    for i in X:
        p1, d1 = graph.dijkstra(i, 1)
        path_X.append(p1)
        distances_X.append(d1)

    # In rj is stored the right part of the query B for every i for evey X
    # Query box = [inf,r1] x ... x [inf,rk]
    rj = []
    temp = []
    for i in separatorList:
        for k in range(0, len(X)):
            for j in separatorList:
                if j < i:
                    temp.append(distances_X[k][i] - distances_X[k][j] - 1)
                else:
                    temp.append(distances_X[k][i] - distances_X[k][j])
            rj.append(temp)
            temp = []

    print()
    # TODO: calculate=> e(x, zi; Y)

    # [E5] Recurse on G[X]
def main():
    test = Graphs()
    g = test.createGrid2D(3, 3)
    edges2 = []

    # Configuring edges to be readable for Dijkstra function
    for i in range(len(g)-2):
        for j in range(len(g[i]) ):
            edges = [i, g[i][j], 1]
            if edges2 == []:
                edges2 = [edges[:]]
            else:
                edges2.append(edges)


    algorithmE(g, separatorList, edges2, X, Y)


if __name__ == "__main__":
    main()