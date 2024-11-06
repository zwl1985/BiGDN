import argparse
import multiprocessing
import statistics
import time

import networkx as nx
import torch

import utils
import models
from agent import get_q_net_input


def read_test_graph(test_graph_path):
    """
    Read the test graph data from the given file path, and calculate and set the 'weight' attribute of each edge to the reciprocal of the node's in degree.

    Parameters:
        Test_graph_path (str): The file path containing the list of graph edges.

    return:
        Test_graph (networkx. DiGraph): Read and process the directed graph.

    """
    test_graph = nx.read_edgelist(test_graph_path, create_using=nx.DiGraph, nodetype=int)
    for u, v, a in test_graph.edges(data=True):
        # Real time calculation of node's in degree here
        in_degree = test_graph.in_degree(v)
        # Set the 'weight' attribute using real-time calculated in degrees
        a['weight'] = 1 / in_degree
    return test_graph


def test(q_net, num_features, test_graph, num_tests, k, device, selection_strategy='o'):
    """
    Use real-time calculation of in degree to set the 'weight' attribute for testing the Q network, and select seed nodes from the graph based on the given selection strategy.

    Parameters:
        Q_net (nn. Module): A Q-network model used to calculate the Q-value of nodes.
        Num_features (int): The number of input features.
        Test_graph (networkx.Graph or similar object): The graph to be tested.
        Num_tests (int): The number of times the test was conducted.
        K (int): The number of seed nodes to be selected.
        Device (torch. device): Used to specify on which device the network is running (e.g. 'cpu' or 'CUDA').
        Selection_strategy (str, optional): Select a strategy, which can be either 'o' (real-time) or 'ob' (interactive). The default is' o '.

    return:
        List: A list containing the selected seed node ID.

    Raises:
        ValueError: If the value of the selection_strategy parameter is not 'o' or 'ob'.
    """
    if selection_strategy == 'o':
        # one-time
        all_t = 0
        for _ in range(num_tests):
            t1 = time.time()
            # Initialize the state of all nodes to 0
            states = torch.zeros(test_graph.number_of_nodes(), dtype=torch.float, device=device)
            # Obtain the input data required for Q network
            data = get_q_net_input([test_graph], num_features, device)
            # Calculate the Q values of all nodes
            q_values = q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
            # Retrieve the k nodes with the highest Q value
            seeds = torch.topk(q_values, k=k).indices.tolist()
            t2 = time.time()
            all_t += t2 - t1
        print(f'one-time：{num_tests}round average duration：{all_t / num_tests}s, seeds：{seeds}', end='\t')
        return seeds

    elif selection_strategy == 'ob':
        # interactive
        all_t = 0
        for _ in range(num_tests):
            t1 = time.time()
            # Initialize the state of all nodes to 0
            states = torch.zeros(test_graph.number_of_nodes(), dtype=torch.float, device=device)
            # Obtain the input data required for Q network
            data = get_q_net_input([test_graph], num_features, device)
            seeds = []
            # Cycle k times to select k nodes
            for _ in range(k):
                # Calculate the Q values of all nodes
                q_values = q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
                # Loop until an unselected node is found
                while True:
                    # Select the node with the highest Q value
                    seed = q_values.argmax().item()
                    # If the node is not selected, add it to the seed set
                    if seed not in seeds:
                        seeds.append(seed)
                        break
                    q_values[seed] = -1e6
                states[seed] = 1.0
            t2 = time.time()
            all_t += t2 - t1
        print(f'one-time：{num_tests}round average duration：{all_t / num_tests}s, seeds：{seeds}', end='\t')
        return seeds

    else:
        raise ValueError(f'The selection_strategy parameter does not have a value of {selection_strategy}.')


def main(args):
    test_graph_path = args.test_graph_path
    num_features = 64
    k = args.k
    num_tests = args.num_tests
    teacher = args.teacher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if teacher:
        q_net = models.QValueNet(num_features).to(device)
        # Load model parameters
        q_net.load_state_dict(torch.load(args.model))
    else:
        q_net = models.StudentQValueNet(num_features).to(device)
        # Load model parameters
        q_net.load_state_dict(torch.load(args.model))

    # Read the test chart and set the edge weights at the same time
    test_graph = read_test_graph(test_graph_path)

    # test
    seeds = test(q_net, num_features, test_graph, num_tests, k, device, args.selection_strategy)

    # Perform R-round propagation
    num_workers = 5
    R = args.R
    with multiprocessing.Pool(num_workers) as pool:
        mc_args = [[test_graph, seeds, int(R / num_workers)] for _ in range(num_workers)]
        results = pool.starmap(utils.computeMC, mc_args)
    reward = statistics.mean(results)
    print(f'spread influence：{reward}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Required parameters for testing")
    parser.add_argument("--test_graph_path", type=str, default="test_graphs/Wiki-2.txt", help="The path of the test chart file.")
    parser.add_argument("--k", type=int, default=10, help="The number of selected seed nodes.")
    parser.add_argument("--model", type=str, default='q_net.pth', help="The trained model parameters used.")
    parser.add_argument("--num_tests", type=int, default=10, help="The number of tests conducted.")
    parser.add_argument("--selection_strategy", type=str, default='o', help="The strategy for selecting seed nodes.")
    parser.add_argument("--R", type=int, default=10000, help="The frequency of dissemination.")
    parser.add_argument("--teacher", action='store_true', help="Use the teacher model.")
    parser.add_argument("--noteacher", action='store_false', dest='teacher', help="Do not use teacher models.")
    args = parser.parse_args()

    main(args)
