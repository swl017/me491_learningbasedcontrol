# -*- coding: utf-8 -*-
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import degree_alg
import numpy as np
import textwrap

from numpy import ma
from numpy.ma.core import maximum_fill_value


def plot_graph(adjacency_matrix, path=None):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    plt.cla()
    fig = plt.figure(1)

    plt.title("Korea highway map")
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    # plot
    labels = nx.get_edge_attributes(G, 'weight')
    pos_map = [
        [223, -85],
        [137, -104],
        [226, -213],
        [262, -269],
        [445, -200],
        [538, -26],
        [155, -245],
        [310, -377],
        [490, -420],
        [180, -480],
        [200, -688],
        [385, -578],
        [550, -500],
        [700, -434],
        [113, -756],
        [357, -752],
        [450, -672],
        [630, -675],
        [684, -572],
    ]
    # city_list = ['서울','인천','평택','천안','제천','강릉','당진','대전','김천','군산','광주','함양','대구','포항','목포','여수','진주','부산','울산']
    city_list = ['Seoul', 'Incheon', 'Pyeongtaek', 'Cheonan', 'Jecheon', 'Gangneung', 'Dangjin', 'Daejeon', 'Gimcheon',
                 'Gunsan',
                 'Gwangju', 'Hamyang', ' Daegu', 'Pohang', 'Mokpo', 'Yeosu', 'Jinju', 'Busan', 'Ulsan']
    city_list = {i: city_list[i] for i in range(len(pos_map))}
    pos = {i: pos_map[i] for i in range(len(pos_map))}
    nx.draw(G, labels=city_list, pos=pos, with_labels=True, font_size=10)
    nodes = nx.draw_networkx_nodes(G, pos, node_color='#8FC2FF')
    nodes.set_edgecolor('white')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=8)

    if path is not None:
        policy_edge = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=policy_edge, edge_color='r', width=2)

    fig.set_size_inches(10, 10)
    plt.gca().set_aspect('equal')
    plt.show()


def print_optim_path(optim_path):
    optim_path_info = ["{}->{}".format(optim_path[i], optim_path[i + 1]) for i in range(len(optim_path) - 1)]
    return ", ".join(optim_path_info)


def get_optim_policy(D=None, optim_value=None, depart_pos=None, terminal_pos=None, gamma=None):
    get_optim_policy = []
    # Todo
    # # Go to every adjacent location -> among 1!+2!+...+11!
    # # filter out links that are not adjacent
    # num_nodes = len(D)
    # adjacency = D[depart_pos]
    # optim_policy_candidate = []
    # if num_nodes > 1:
    #     for i in range(1, num_nodes): # when terminal pos is the at the ith
    #         for j in range(0, i): # search 0~i-1th action
    #             # find adjacent node
    #             last_pos = depart_pos
    #             actions = []
    #             for k in range(0, num_nodes):
    #                 if k is not last_pos and adjacency[k] is not 0:
    #                     # add to the que
    #                     action = [k] 

    #                 optim_policy_candidate += []

    max_action_length = 1000
    num_nodes = len(D)
    gamma = 0.9
    all_actions = []
    all_values = [] # value
    for i in range(0, max_action_length): # when terminal pos is the at the ith
        last_pos = depart_pos
        action_list = []
        escape = False
        for ith_action in range(0,num_nodes):
            for ith_action_slot in range(0, i): # search i numbers of actions
                reward = -D[last_pos][ith_action]
                if ith_action is not last_pos and reward is not 0:
                    # print(ith_action, last_pos, D[last_pos][ith_action])
                    # add to the que if it is a valid action
                    action_list += [ith_action]
                    if ith_action_slot is i-1 and ith_action is terminal_pos:
                        all_actions += [action_list]
                        value = 0
                        last_action = depart_pos
                        for action in action_list:
                            reward = D[last_action][action]
                            value += reward + gamma*value
                        all_values += [value]
                    last_pos = ith_action
                    # print(all_values)
                else:
                    escape = True
                    break
            if escape:
                break
    
    optim_ind = np.argmax(all_values)
    optim_value = all_values[optim_ind]
    optim_policy = all_actions[optim_ind]
    get_optim_policy = optim_policy
    return get_optim_policy

def get_all_actions(num_nodes, i, D, action_list, terminal_pos, all_actions, depart_pos, gamma, all_values):
    for ith_action in range(0,num_nodes):
        for ith_action_slot in range(0, i): # search i numbers of actions
            if D[last_pos][ith_action] is not 0:
                print(ith_action, last_pos, D[last_pos][ith_action])
            if ith_action is not last_pos and D[last_pos][ith_action] is not 0:
                # add to the que if it is a valid action
                action_list += [ith_action]
                if ith_action_slot is i and ith_action is terminal_pos:
                    all_actions += [action_list]
                    value = 0
                    last_action = depart_pos
                    for action in action_list:
                        reward = D[last_action][action]
                        value += reward + gamma*value
                    all_values += [value]
                last_pos = ith_action
    return all_actions

def get_optim_path(D=None, optim_value=None, depart_pos=None, terminal_pos=None, gamma=None):
    optim_path = []
    # Todo
    # Find all path from A to B
    # Get value for all the path
    # Pick highest value
    return optim_path


def get_optim_value(D=None, threshold=0.001, gamma=0.9, depart_pos=7, terminal_pos=0):
    optim_value = []
    # Todo

    return optim_value


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='prob1_value_iter.py',
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent('''\
        City List : 
         'Seoul',         0
         'Incheon',       1
         'Pyeongtaek',    2
         'Cheonan',       3
         'Jecheon',       4
         'Gangneung',     5
         'Dangjin',       6
         'Daejeon',       7
         'Gimcheon',      8
         'Gunsan',        9
         'Gwangju',       10
         'Hamyang',       11
         'Daegu',         12
         'Pohang',        13
         'Mokpo',         14
         'Yeosu',         15
         'Jinju',         16
         'Busan',         17
         'Ulsan',         18
         ''')
    )

    parser.add_argument("-d", "--depart", help="departing city(default Daejeon)", type=str, default="7")
    parser.add_argument("-t", "--terminal", help="terminalal city(default Seoul)", type=str, default="0")

    args = parser.parse_args()

    D = np.genfromtxt('HW1_adjacency_matrix.csv', delimiter=',')
    D = D.astype(int)

    num_nodes = len(D)
    depart_pos = int(args.depart)
    terminal_pos = int(args.terminal)
    gamma = 0.9

    optim_value = get_optim_value(D, threshold=0.001, gamma=gamma, depart_pos=depart_pos, terminal_pos=terminal_pos)
    optim_policy = get_optim_policy(D, optim_value, depart_pos, terminal_pos, gamma)
    optim_path = get_optim_path(D, optim_value, depart_pos, terminal_pos, gamma)

    print("-" * 20)
    print("The value of states using value_iteration")
    print("{}".format(list(np.around(optim_value, decimals=2))))
    print("-" * 20)
    print("The best action for every node")
    print(optim_policy)
    print("-" * 20)
    print("The best action from departure to the terminal")
    print(optim_path)
    print(print_optim_path(optim_path))

    plot_graph(D, path=optim_path)
