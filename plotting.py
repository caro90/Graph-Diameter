import matplotlib.pyplot as plt
import csv
import networkx as nx

def main():

    # time = list()
    # nodes = list()
    #
    # with open('timings.csv', mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count > 0:
    #             nodes.append(row["num of nodes"])
    #             time.append(row["time"])
    #         line_count += 1
    #     print(f'Processed {line_count} lines.')
    #
    # plt.plot(nodes, time)
    # plt.loglog(nodes, time)
    # plt.show()

    G = nx.binomial_graph(100, 0.05)
    G = nx.erdos_renyi_graph(50, 0.1)
    G = nx.connected_caveman_graph(30, 5)
    G = nx.erdos_renyi_graph(50, 0.02)
    #G = nx.gnm_random_graph(5, 5, seed=2, directed=False)
    flag = 1
    if flag == 1:
        # print the adjacency list
        #   for line in nx.generate_adjlist(p[1]):
        #       print(line)
        # write edgelist to grid.edgelist
        nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
        # read edgelist from grid.edgelist
        H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

        nx.draw(H, with_labels=True)
        plt.show()


if __name__ == "__main__":
    main()