import pandas as pd
import numpy as np
import networkx as nx
from typing import List
import seaborn as sns
import glob
import random
import matplotlib.pyplot as plt
import csv
import json
from operator import itemgetter
from networkx.algorithms import community
from networkx.algorithms.structuralholes import effective_size
from networkx.algorithms.flow import shortest_augmenting_path
from util import set_edge_community, set_node_community, get_color, get_actual_name
from mp1_plots import plot_degree_boxplot, plot_centrality, plot_edge_connectivity, plot_six_degrees_study, \
    plot_ego_of_largest, ego_graph_simple, plot_ego_communities, plot_number_of_communities
import multiprocessing.pool


def get_networkx_graph(file_name) -> nx.Graph:
    graph_data = pd.read_csv(file_name)
    graph_type = nx.Graph()

    return nx.from_pandas_edgelist(graph_data, create_using=graph_type)


def convert_to_csv():
    cond_fd = open('../ca-HepTh.txt')
    new_data_fd = open('./hepth.csv', 'w')
    new_csv_writer = csv.DictWriter(new_data_fd, fieldnames=["source", "target"])
    all_collab_list = []
    for data in cond_fd:
        split = data.split()
        if len(split) <= 2:
            temp_dict = {
                "source": split[0],
                "target": split[1]
            }
            all_collab_list.append(temp_dict)

    new_csv_writer.writeheader()
    new_csv_writer.writerows(all_collab_list)


def get_shortest_paths(item):
    node_list = list(item["graph"].nodes)
    path_dict_list: List = list()
    key_list = []

    for i in range(100000):
        if i % 100 == 0:
            print(f"Finished {i + 1} iterations for {item['name']}")
        rand_1 = random.choices(node_list, k=2)
        rand_text = f"{rand_1[0]}_{rand_1[1]}"
        if rand_text in key_list:
            print(f"Skipped {rand_text}")
        else:
            try:
                path = nx.shortest_path(item["graph"], source=rand_1[0], target=rand_1[1])
                temp_dict = {
                    rand_text: len(path)
                }
                key_list.append(rand_text)
                path_dict_list.append(temp_dict)
            except nx.NetworkXNoPath:
                temp_dict = {
                    rand_text: -1
                }
                key_list.append(rand_text)
                path_dict_list.append(temp_dict)

    json_file = open(f"./results/path/{item['name']}.json", "w")
    json.dump(path_dict_list, json_file)


def get_five_number_summary(graph_dict_list):
    for item in graph_dict_list:
        degrees = item["graph"].degree
        deg_values = [deg[1] for deg in degrees]
        quartiles = np.percentile(deg_values, [25, 50, 75])
        # calculate min/max
        data_min, data_max = min(deg_values), max(deg_values)
        # print 5-number summary
        print(f"Name of Graph; {item['name']}")
        print('Min: %.3f' % data_min)
        print('Q1: %.3f' % quartiles[0])
        print('Median: %.3f' % quartiles[1])
        print('Q3: %.3f' % quartiles[2])
        print('Max: %.3f' % data_max)


def get_beweenness_centrality(item):
    print(f"Processing {item['name']}")
    centrality = nx.betweenness_centrality(item["graph"])
    json_fd = open(f"./results/centrality/{item['name']}.json", "w")
    json.dump(centrality, json_fd)


def get_connectivity_analysis(item):

    edge_connectivity_list = []
    key_list = []
    for i in range(1000):
        if i % 100 == 0:
            print(f"Finished {i} of file: {item['name']}")
        temp_dict = dict()
        nodes = list(item["graph"].nodes)
        choices = random.choices(nodes, k=2)
        nodes_ = f"{choices[0]}_{choices[1]}"
        if nodes_ in key_list:
            continue
        else:
            val = nx.edge_connectivity(item["graph"], choices[0], choices[1])
            temp_dict[nodes_] = val
            edge_connectivity_list.append(temp_dict)
            key_list.append(nodes_)
    json_file = f"./results/connectivity/{item['name']}.json"
    json_fd = open(json_file, "w")
    json.dump(obj=edge_connectivity_list, fp=json_fd)


def get_basic_info(graph_dict_list):
    for item in graph_dict_list:
        print(item["name"])
        print(nx.info(item["graph"]))


def run():
    files_list = glob.glob("./dataset/*.csv")
    graph_dict_list = []
    for file_ in files_list:
        actual_name = get_actual_name(file_)
        temp_dict = {
            "name": actual_name,
            "file": file_,
            "graph": get_networkx_graph(file_)
        }
        graph_dict_list.append(temp_dict)

    get_basic_info(graph_dict_list)

    get_five_number_summary(graph_dict_list)
    plot_degree_boxplot(graph_dict_list)
    plot_centrality()
    plot_edge_connectivity()
    plot_six_degrees_study()
    get_connectivity_analysis(graph_dict_list)
    plot_ego_of_largest(graph_dict_list)
    ego_graph_simple(graph_dict_list)
    plot_ego_communities(graph_dict_list)
    plot_number_of_communities()
    with multiprocessing.Pool(5) as p:
        p.map(get_connectivity_analysis, graph_dict_list)

    with multiprocessing.Pool(5) as p:
        p.map(get_beweenness_centrality, graph_dict_list)

    with multiprocessing.Pool(5) as p:
        p.map(get_shortest_paths, graph_dict_list)


if __name__ == '__main__':
    run()
