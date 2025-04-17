import multiprocessing
import random
import statistics

import utils


class GraphEnvironment:
    def __init__(self, graphs, k, gamma=0.99, n_steps=2, method='MC', R=10000, num_workers=8):
        """
        G: The graph of networkx, Graph or DiGraph;
        k: Seed set size;
        n_steps: the step size used to calculate rewards;
        method: Method for calculating rewards;
        R: Use Monte Carlo to estimate the number of reward rounds;
        numw_workers: How many cores are used to calculate propagation range
        """
        self.graphs = graphs  # graph List
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.method = method
        self.R = R
        self.num_workers = num_workers
        self.graph = None  # The currently used graph
        # Current status, each position represents whether a node has been selected, with 1 indicating selected and 0 indicating unselected
        self.state = None
        # Reward for the previous state
        self.preview_reward = 0
        # Record the status, actions, rewards, and next steps of each exploration to calculate the reward for n steps
        self.states = []
        self.actions = []
        self.rewards = []
        # self.next_states = []

        self.seeds = []
        self.state_records = {}

    def reset(self):
        """
        Reset the environment.
        """
        self.graph = random.choice(self.graphs)
        self.seeds = []
        self.state = [0] * self.graph.number_of_nodes()
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        # self.next_states = []
        return self.state

    def step(self, action):
        """
        Transfer to a new state based on the given action.
        """
        self.states.append(self.state.copy())
        self.state[action] = 1
        self.seeds.append(action)
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        if done:
            self.states.append(self.state.copy())

        self.actions.append(action)
        self.rewards.append(reward)
        # self.next_states.append(self.state)
        return reward, self.state, done

    def compute_reward(self):
        str_seeds = str(id(self.graph)) + str(sorted(self.seeds))
        if self.method == 'MC':
            if str_seeds in self.state_records:
                current_reward = self.state_records[str_seeds]
            else:
                with multiprocessing.Pool(self.num_workers) as pool:
                    args = [[self.graph, self.seeds, int(self.R / self.num_workers)] for _ in
                            range(self.num_workers)]
                    results = pool.starmap(utils.computeMC, args)
                current_reward = statistics.mean(results)
            r = current_reward - self.preview_reward
            self.preview_reward = current_reward
            self.state_records[str_seeds] = current_reward
            return r
        else:
            pass

    def n_step_add_buffer(self, buffer):
        states = self.states
        rewards = self.rewards
        n = self.n_steps
        gamma = self.gamma
        
        # Directly limit the cycle range and avoid processing the situation of insufficient n steps
        for i in range(len(states) - n):
            # Determines whether it is terminated
            done = (i + n) == (len(states) - 1)
            next_state = states[i + n]
            
            # Calculate the n-step reward
            n_reward = sum(rewards[i + j] * (gamma ** j) for j in range(n))
            
            buffer.add(states[i], self.actions[i], n_reward, next_state, terminal, self.graph)
