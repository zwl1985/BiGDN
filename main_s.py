import argparse
import os
import sys
from datetime import datetime

import networkx as nx
import torch
from tqdm import tqdm

import utils
from agent import StudentAgent
from environment import GraphEnvironment
from models import QValueNet


def explore(env, agent, eps, replay_buffer, num_episodes, train=True, show_bar=True):
    """
    Explore using a given agent in a given environment.

    Parameters:
        Env (object): Environment object used to interact with agents.
        Agent (object): Agent object, including policy network, epsilon value, etc.
        EPS (float): The epsilon value of the current round, used to balance exploration and utilization.
        Replay_buffer (object): An experience replay buffer used to store data generated during the exploration process.
        Num_ episodes (int): The number of sequences to explore.
        Train (boolean, optional): Whether in training mode, defaults to True. In training mode, data will be added to the playback buffer.
        Show_mar (boolean, optional): Whether to display a progress bar, default to True.

    return:
        If train is False, return an episode return of float type, which is the accumulated reward during the testing process;
        Otherwise, no value will be returned (None).
    """

    agent.epsilon = eps

    if train:
        if show_bar:
            bar = tqdm(total=num_episodes, desc=f'epsilon={eps}时探索{num_episodes}条序列')  # 初始化进度条

        for _ in range(num_episodes):
            state = env.reset()  # Reset the environment to obtain the initial state
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env)  # Select action based on current state and epsilon value
                reward, next_state, done = env.step(action)  # Execute actions, receive rewards, and move on to the next state
                episode_return += reward
                state = next_state

            # After the exploration sequence is completed, add the n-step reward to the experience replay buffer
            env.n_step_add_buffer(replay_buffer)

            if show_bar:
                bar.update(1)

        if show_bar:
            bar.close()

    else:
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env)
                reward, next_state, done = env.step(action)
                episode_return += reward
                state = next_state

        return episode_return


def train(agent, num_epochs, train_graphs, env, test_env, replay_buffer, batch_size, folder_path):
    """
    Train reinforcement learning agents on a given dataset.

    Parameters:
        Agent: A proxy object that includes training components such as Q networks.
        Num_ epochs (int): The total number of training epochs.
        Train_graphs (list): A list of subgraphs used for training.
        Env (object): Training environment object.
        Test_dev (GraphEnvironment): Test environment object.
        Replay_buffer: An experience replay buffer object used to store training data.
        Batch_2 (int): The batch size sampled from the playback buffer each time.
        Folderpath (str): The folder path used to store Q network parameters.

    return:
        No return value, but will update the Q network parameters of the agent and save some intermediate results.
    """

    # In the initial exploration phase, fill the experience replay buffer
    explore(env, agent, 1, replay_buffer, 1)

    # Set the starting and ending values for exploration rate
    eps_start = 1
    eps_end = 0.05

    tbar = tqdm(total=num_epochs, desc=f'使用{len(train_graphs)}个子图训练')

    for i in range(num_epochs):
        eps = eps_end + max(0.0, (eps_start - eps_end) * (num_epochs // 2 - i) / (num_epochs // 2))

        if i % 10 == 0:
            # Save the parameters of Q network
            torch.save(agent.q_net.state_dict(), f'{folder_path}/q_net_{i // 10}.pth')

            # Explore using the current exploration rate EPS and fill the experience replay buffer
            explore(env, agent, eps, replay_buffer, 1, True, False)

            # Test on the testing environment and receive rewards
            rewards = explore(test_env, agent, 0, None, 1, False, False)

            tbar.write(
                f'{i}/{num_epochs}：种子集：{test_env.seeds}，奖励：{rewards}，回放缓冲区长度：{len(replay_buffer)}', file=sys.stdout)

        # Sample data from the experience replay buffer and update the Q network
        states, actions, rewards, next_states, dones, graphs = replay_buffer.sample(batch_size)
        agent.update(states, actions, rewards, next_states, graphs, dones)

        tbar.update(1)


def main(args):
    lr = args.lr
    gamma = args.gamma
    epsilon = 1
    target_update = args.target_update
    buffer_size = args.buffer_size
    batch_size = 16
    num_epochs = args.num_epochs
    k = args.k
    n_steps = args.n_steps
    num_features = args.num_features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = utils.ReplayBuffer(buffer_size)

    teacher_q_net = QValueNet(num_features).to(device)
    teacher_q_net.load_state_dict(torch.load('q_net.pth'))

    agent = StudentAgent(num_features, gamma, epsilon, lr, device, teacher_q_net, target_update=target_update,
                         n_steps=n_steps)

    # Create a folder to save the model
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{time_str}"
    folder_path = os.path.join(os.getcwd(), 'my_model', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    g_folder_path = 'train_graphs'
    filenames = os.listdir(g_folder_path)
    train_graphs = []
    test_graphs = []
    for filename in filenames:
        if filename == '.ipynb_checkpoints':
            continue
        file_path = os.path.join(g_folder_path, filename)
        g = nx.read_edgelist(file_path, nodetype=int, create_using=nx.DiGraph)
        for u, v, a in g.edges(data=True):
            a['weight'] = 1 / len(list(g.predecessors(v)))
        if 'graph1.txt' == filename:
            test_graphs.append(g)
        train_graphs.append(g)

    env = GraphEnvironment(train_graphs, k, gamma, n_steps, R=10000, num_workers=1)
    test_env = GraphEnvironment(test_graphs, k, gamma, n_steps, R=10000, num_workers=1)

    train(agent, num_epochs, train_graphs, env, test_env, replay_buffer, batch_size, folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters required for student model training")
    parser.add_argument("--lr", type=float, default="0.001", help="Learning rate.")
    parser.add_argument("--k", type=int, default=5, help="The number of selected seed nodes.")
    parser.add_argument("--n_steps", type=int, default=1, help="Step size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--target_update", type=int, default=100, help="The frequency of target network updates.")
    parser.add_argument("--buffer_size", type=int, default=50000, help="The size of the experience replay pool.")
    parser.add_argument("--num_features", type=int, default=64, help="Feature dimension.")
    parser.add_argument("--num_epochs", type=int, default=20000, help="The number of training rounds.")
    parser.add_argument("--R", type=int, default=10000, help="The frequency of dissemination.")
    args = parser.parse_args()

    main(args)
