# -*- coding: utf-8 -*-
from argparse import ArgumentParser, RawTextHelpFormatter
from operator import le
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import textwrap

import copy

from numpy.core.fromnumeric import argmax

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
    epsilon = 1e-3 # convergence threshold
    policy_converge = False
    value_converge  = False
    # policy_k  = np.ones(len(D[0]))
    policy_k  = np.array([1,2,3,4,5,4,7,8,7,10,11,12,13,12,15,16,17,18,17]) # Go to next city if possible
    policy_k1 = np.zeros(len(policy_k))
    policy_implicit = np.array([1,2,3,4,5,4,7,8,7,10,11,12,13,12,15,16,17,18,17])
    value_k   = np.array(D[0]) * (0) # v_k
    value_k1  = np.array(D[0]) * (0) # v_(k+1)
    reward    = np.array(D) * (-1)
    # Until policy converges
    while not policy_converge:
        # Until value converges
        while not value_converge:
            value_k1, policy_implicit = value_iter(value_k, value_k1, reward, policy_implicit, gamma)
            # value_k1   = evaluate_policy(value_k, value_k1, policy_k, reward, gamma)
            value_diff = value_k1 - value_k
            value_max_diff = -1
            for i in range(len(value_k)):
                if value_k[i] != 0 and abs(value_diff[i]) > value_max_diff:
                    value_max_diff = copy.deepcopy(abs(value_diff[i]))
            if value_max_diff < epsilon and value_max_diff != -1: # Teminate when value converges
                value_converge = True
            else:
                value_k = copy.deepcopy(value_k1) # else update value
        policy_diff = policy_k1 - policy_k
        policy_max_diff = -1
        for i in range(len(policy_k)):
            if policy_k[i] !=0 and abs(policy_diff[i]) > policy_max_diff:
                policy_max_diff = copy.deepcopy(abs(policy_diff[i]))
        if  policy_max_diff < epsilon and policy_max_diff != -1:
            policy_converge = True
        else:
            policy_k1 = improve_policy(value_k, policy_k, policy_k1, reward, gamma)
        policy_k = copy.deepcopy(policy_k1)

    print("Implicit optimal policy by value iteration: ")
    print(policy_implicit)
    optim_value = value_k
    get_optim_policy = policy_k

    return get_optim_policy

def value_iter(value_k, value_k1, reward, policy_implicit, gamma):
    # Single iteration of value iter
    # For every states
    for s in range(len(value_k)):
        max_value = -9999
        argmax_action = 0
        # For every next states s_
        for s_ in range(len(value_k)):
            if s_ is not s: # Cannot stay the same place
                if reward[s][s_] != 0: # Exclude actions that are impossible
                    value  = reward[s][s_] + gamma * value_k[s_]
                    action = s_
                    if value > max_value:
                        max_value     = value
                        argmax_action = action
        value_k1[s]        = max_value
        policy_implicit[s] = argmax_action

    return value_k1, policy_implicit

def evaluate_policy(value_k, value_k1, policy_k, reward, gamma):
    # Evaluate current policy given current value table
    # For every states
    for s in range(len(policy_k)):
        s_ = policy_k[s] # next state s_
        if reward[s][s_] != 0: # Exclude actions that are impossible
            value_k1[s] = reward[s][s_] + gamma * value_k[s_]

    return value_k1

def improve_policy(value_k, policy_k, policy_k1, reward, gamma):
    print("Performing policy improvement")
    # Update current policy according to policy evaluation
    for s in range(len(value_k)):
        max_value     = -9999
        argmax_action = 0
        for s_ in range(len(value_k)):
            Rt1 = reward[s][s_]
            if Rt1 != 0: # Exclude actions that are impossible
                value = Rt1 + gamma * value_k[s_]
                if value > max_value:
                    max_value     = value
                    argmax_action = s_
        policy_k1[s] = argmax_action

    return policy_k1

def get_optim_path(D=None, optim_value=None, depart_pos=None, terminal_pos=None, gamma=None):
    optim_path = []
    # Todo
    return optim_path


def get_optim_value(D=None, threshold=0.001, gamma=0.9, depart_pos=7, terminal_pos=0):
    optim_value = []
    # Todo
    return optim_value


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='prob2_policy_iter.py',
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

    D = np.genfromtxt('HW2_adjacency_matrix.csv', delimiter=',')
    D = D.astype(int)

    num_nodes = len(D)
    depart_pos = int(args.depart)
    terminal_pos = int(args.terminal)
    gamma = 0.9

    optim_value = get_optim_value(D, threshold=0.001, gamma=gamma, depart_pos=depart_pos, terminal_pos=terminal_pos)
    optim_policy = get_optim_policy(D, optim_value, depart_pos, terminal_pos, gamma)
    optim_path = get_optim_path(D, optim_value, depart_pos, terminal_pos, gamma)

    print("-" * 20)
    print("The value of states using policy_iteration")
    print("{}".format(list(np.around(optim_value, decimals=2))))
    print("-" * 20)
    print("The best action for every node")
    print(optim_policy)
    print("-" * 20)
    print("The best action from departure to the terminal")
    print(optim_path)
    print(print_optim_path(optim_path))

    plot_graph(D, path=optim_path)
