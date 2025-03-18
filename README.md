# BiGDN

Implementation of "BiGDN: An End-To-End Influence Maximization Framework Based on Deep Reinforcement Learning and Graph Neural Networks"

Run the code
------------

#### Train BiGDN-Teacher model

	python main.py 

#### Train BiGDN-Student model

	python main_s.py 

#### Test BiGDN-Teacher model

	python test.py --test_graph_path test_data/Wiki-2.txt \
                     --teacher\
                     --model q_net.pth \
                     --k 10 
                     --selection_strategy o

#### Test BiGDN-Student model

	python test.py --test_graph_path test_data/Wiki-2.txt \
                     --noteacher \
                     --model q_net_s.pth \
                     --k 10\
                     --selection_strategy o


Dependency requirement
----------------------

- Python 3.9.18
- NumPy 1.23.5
- PyTorch 1.9.1+cu111
- PyG (PyTorch Geometric) 2.4.0
- PyTorch Scatter 2.0.7
- Tqdm 4.66.1
- networkx 3.2.1

Code files
----------

- main.py: load program arguments, graphs and set up RL agent and environment and conduct simulation, train RL agent(teacher).
- main_s.py: load program arguments, graphs and set up RL agent and environment and conduct simulation, train RL agent(student).
- test.py: test RL agent.
- models.py: define parameters and structures of BiGDN and BiGDNs.  
- agent.py: define agents to follow reinforcement learning procedure.
- environment.py: store the process of simulation.  
- utils.py: obtain graph information function and use Monte Carlo simulation of propagation range under IC model and ReplayBuffer.
- node_encoder/train.ipynb: implementation and Pre training of Node Encoder.
- graph_process.ipynb: process the graph by converting nodes into consecutive integers numbered from 0.

Please refer to the code for specific parameter settings and implementation.

Folders
----------

- test_graphs: there are 8 test graphs, the YouTube graph file is too large to upload.
- train_graphs: there are 15 training graphs, consistent with the training graphs of ToupleGDD.
- node_encoder: contains training graphs and training code used by NE.

License
-------
This project is licensed under the terms of the [MIT](LICENSE) license.

Cite
-------
If the code is used, please cite it:

```text
@article{BiGDN,
title = {BiGDN: An end-to-end influence maximization framework based on deep reinforcement learning and graph neural networks},
journal = {Expert Systems with Applications},
volume = {270},
pages = {126384},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126384},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425000065},
author = {Wenlong Zhu and Kaijing Zhang and Jiahui Zhong and Chengle Hou and Jie Ji},
}
```
