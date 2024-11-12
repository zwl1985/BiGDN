import random
import collections
from collections import deque


def computeMC(graph, S, R):
    """
    compute expected influence using MC under IC
        R: number of trials
    """
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.neighbors(curr_node) \
                                       if
                                       not (child in source_set) and random.random() <= graph.edges[curr_node, child][
                                           'weight'])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)

    return inf / R


def get_graph_edge_index(graph, weight=True):
    """
        Obtain the edge_index and edge_ceight of the graph
    """
    edges = graph.edges()
    sources, targets = zip(*edges)
    edge_index = [sources, targets]

    if weight:
        edges_data = graph.edges(data=True)
        weights = [d['weight'] for u, v, d in edges_data]  
        return edge_index, weights
    else:
        return edge_index


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, graph):
        self.buffer.append((state, action, reward, next_state, done, graph))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, graph = zip(*transitions)
        return state, action, reward, next_state, done, graph

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    pass
