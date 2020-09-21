import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
from networkx.algorithms import community
from operator import itemgetter
from util import get_actual_name, set_node_community, set_edge_community, get_color


def plot_degree_boxplot(graph_dict_list):
    data = []
    xticklabels = []
    for item in graph_dict_list:
        degrees = item["graph"].degree
        deg_value = [deg[1] for deg in degrees]
        data.append(deg_value)
        xticklabels.append(item["name"])
    fig, ax = plt.subplots()
    size = 25
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.yscale("log")
    # build a box plot
    ax.boxplot(data)

    # title and axis labels
    ax.set_title('Boxplots of Degree Distributions for Multiple Collaboration Networks')
    ax.set_xlabel('Name of the Network Being Considered')
    ax.set_ylabel('Count of the degrees [LOG]')
    ax.set_xticklabels(xticklabels)

    # add horizontal grid lines
    ax.yaxis.grid(True)

    # show the plot
    plt.show()


def plot_centrality():
    files = glob.glob("./results/centrality/*.json")
    data = []
    xticklabels = []

    for item in files:
        json_fd = open(item)
        dataset = json.load(json_fd)
        orted_betweenness = sorted(dataset.items(), key=itemgetter(1), reverse=True)
        deg_value = [deg[1] for deg in orted_betweenness]
        data.append(deg_value)
        act_name = get_actual_name(item)
        xticklabels.append(act_name)
    xticklabels.insert(0, "n")
    fig, ax = plt.subplots()
    plt.yscale("log")
    size = 25
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    # build a box plot
    ax.violinplot(data)

    # title and axis labels
    ax.set_title('Violin Plot of Degree Distributions for Multiple Collaboration Networks')
    ax.set_xlabel('Name of the Network Being Considered')
    ax.set_ylabel('Value of Centrality [LOG]')
    ax.set_xticklabels(xticklabels)

    # add horizontal grid lines
    ax.yaxis.grid(True)

    # show the plot
    plt.show()
    plt.close()


def plot_edge_connectivity():
    files = glob.glob("./results/connectivity/*.json")
    data = []
    xticklabels = []

    for item in files:
        json_fd = open(item)
        dataset = json.load(json_fd)
        data_dict = dict()
        for val in dataset:
            data_dict.update(val)
        connectivity = sorted(data_dict.items(), key=itemgetter(1), reverse=True)
        deg_value = [deg[1] for deg in connectivity]
        data.append(deg_value)
        act_name = get_actual_name(item)
        xticklabels.append(act_name)
    xticklabels.insert(0, "n")
    fig, ax = plt.subplots()
    size = 25
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    # build a box plot
    ax.violinplot(data)

    # title and axis labels
    ax.set_title('Violin Plot of Edge Centrality for Multiple Collaboration Networks')
    ax.set_xlabel('Name of the Network Being Considered')
    ax.set_ylabel('Value of Edge Centrality [LOG]')
    ax.set_xticklabels(xticklabels)

    # add horizontal grid lines
    ax.yaxis.grid(True)

    # show the plot
    plt.show()
    plt.close()


def plot_six_degrees_study():
    files = glob.glob("./results/path/*.json")
    data = []
    xticklabels = []
    size = 25
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    for item in files:
        json_fd = open(item)
        dataset = json.load(json_fd)
        data_dict = dict()
        for val in dataset:
            data_dict.update(val)
        connectivity = sorted(data_dict.items(), key=itemgetter(1), reverse=True)
        deg_value = [deg[1] for deg in connectivity]
        data.append(deg_value)
        act_name = get_actual_name(item)
        xticklabels.append(act_name)
        plt.hist(deg_value, alpha=0.3, label=act_name)
    plt.title("Histogram of the Six Degree Study for Each Collaboration Network")
    plt.xlabel("Number of Paths Between Two Nodes")
    plt.ylabel("Count of the paths")
    plt.legend(loc="upper right")
    plt.show()


def plot_ego_of_largest(graph_dict_list):
    for item in graph_dict_list:
        for i in [-1, -2, -3]:
            node_and_degree = item["graph"].degree()
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[i]

            # Create ego graph of main hub
            hub_ego = nx.ego_graph(item["graph"], largest_hub)
            # Draw graph
            pos = nx.spring_layout(hub_ego)
            nx.draw(hub_ego, pos, node_color="b", node_size=50, with_labels=False)

            # Draw ego as large and red
            options = {"node_size": 300, "node_color": "r"}
            nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], **options)

            if i == -1:
                print(f"Ego Network of Largest Node in {item['name']}")
            if i == -2:
                print(f"Ego Network of Second largest Node in {item['name']}")
            if i == -3:
                print(f"Ego Network of Third Largest Node in {item['name']}")
            plt.show()
            plt.close()


def ego_graph_simple(graph_dict_list):

    for item in graph_dict_list:
        for i in [-1, -2, -3]:
            node_and_degree = item["graph"].degree()
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[i]
            print(largest_hub, item["name"])
            hub_ego = nx.ego_graph(item["graph"], largest_hub)

            pos = nx.spring_layout(hub_ego, k=0.1)
            plt.rcParams.update({'figure.figsize': (15, 10)})
            nx.draw_networkx(
                hub_ego,
                pos=pos,
                node_size=0,
                edge_color="#444444",
                alpha=0.1,
                with_labels=False)

            if i == -1:
                print(f"Ego Network of Largest Node in {item['name']}")
            if i == -2:
                print(f"Ego Network of Second largest Node in {item['name']}")
            if i == -3:
                print(f"Ego Network of Third Largest Node in {item['name']}")
            plt.show()
            plt.close()


def plot_ego_communities(graph_dict_list):

    for item in graph_dict_list:
        graph_stat_dict_list = []
        for i in [-1, -2, -3, -4, -5, -6, -7]:
            node_and_degree = item["graph"].degree()
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[i]
            hub_ego = nx.ego_graph(item["graph"], largest_hub)
            temp_dict = {
                "pos": (i*-1),
                "largest_hub": largest_hub,
                "degree": degree
            }
            pos = nx.spring_layout(hub_ego, k=0.1)
            plt.rcParams.update({'figure.figsize': (15, 10)})
            nx.draw_networkx(
                hub_ego,
                pos=pos,
                node_size=0,
                edge_color="#444444",
                alpha=0.05,
                with_labels=False)

            communities = sorted(community.greedy_modularity_communities(hub_ego), key=len, reverse=True)
            temp_dict["communities"] = len(communities)

            graph_stat_dict_list.append(temp_dict)
            set_node_community(hub_ego, communities)
            set_edge_community(hub_ego)

            external = [(v, w) for v, w in hub_ego.edges if hub_ego.edges[v, w]['community'] == 0]
            internal = [(v, w) for v, w in hub_ego.edges if hub_ego.edges[v, w]['community'] > 0]
            internal_color = ["yellow" for e in internal]
            node_color = [get_color(hub_ego.nodes[v]['community']) for v in hub_ego.nodes]
            # external edges
            nx.draw_networkx(
                hub_ego,
                pos=pos,
                node_size=0,
                edgelist=external,
                edge_color="blue",
                node_color=node_color,
                alpha=0.2,
                with_labels=False)
            # internal edges
            nx.draw_networkx(
                hub_ego, pos=pos,
                edgelist=internal,
                edge_color=internal_color,
                node_color=node_color,
                alpha=0.05,
                with_labels=False)
            print(f"Ego network for {item['name']} and {i}")
            plt.title(f"Ego network {largest_hub} of {item['name']}")
            plt.show()
            plt.close()

        json_fd = open(f"./results/communities/{item['name']}.json", "w")
        json.dump(graph_stat_dict_list, json_fd)


def plot_number_of_communities():
    files = glob.glob("./results/communities/*.json")
    data = []
    xticklabels = []

    for item in files:
        json_fd = open(item)
        dataset = json.load(json_fd)
        data_dict = dict()
        curr_ranks = []
        for val in dataset:
            curr_ranks.append((val["communities"], val["pos"]))

        connectivity = sorted(curr_ranks, key=itemgetter(1), reverse=True)
        print(connectivity)
        deg_value = [deg[0] for deg in connectivity]
        deg_value = list(reversed(deg_value))
        data.append(deg_value)
        act_name = get_actual_name(item)
        print(connectivity, act_name)
        xticklabels.append(act_name)

    size=25
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)

    color_list = ['b', 'g', 'r', 'k', 'm']
    gap = .8 / len(data)
    for i, row in enumerate(data):
        X = np.arange(len(row))
        plt.bar(X + i * gap, row,
                width=gap,
                color=color_list[i % len(color_list)], alpha=0.8)
        plt.xticks(range(7), ["Degree 1", "Degree 2", "Degree 3", "Degree 4", "Degree 5", "Degree 6", "Degree 7"])

    plt.title("Community Count for Top 7 Nodes With Largest Degree in Each Network")
    plt.ylabel("Count of communities")
    plt.xlabel("Degree i refers to the ith highest Degree in that Network")
    plt.legend(xticklabels)
    plt.show()
