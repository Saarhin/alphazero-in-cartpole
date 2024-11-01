import torch.nn as nn
import graphviz
import numpy as np
import torch


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def trans_coordinate(coordinate: list, grid_size: int, out_type: str) -> list:
    """
    the original point is the left upper corner [0, 0] in computer science coordinate systems;
    the original point is the left lower corner [0, 5] in VTR coordinate systems.
    """
    if out_type == "cs":
        return [grid_size - 1 - coordinate[1], coordinate[0]]
    elif out_type == "vtr":
        return [coordinate[1], grid_size - 1 - coordinate[0]]


def fill_place_file(
    results,
    grid_size,
    file_path=None,
):
    new_lines = list()
    with open(file_path, "r") as file:
        results_dict = dict()
        for index, line in enumerate(file.readlines()):
            if index >= 5:
                result = trans_coordinate(results[index - 5], grid_size, "vtr")

                if str(result) in results_dict.keys():
                    results_dict[str(result)] += 1
                else:
                    results_dict[str(result)] = 1

                line_split = line.strip().split("\t")
                line_split[2] = str(result[0])
                line_split[3] = str(result[1])
                line_split[4] = str(results_dict[str(result)] - 1)
                new_lines.append("\t".join(line_split) + "\n")

            elif index < 5:
                new_lines.append(line)

    with open(file_path, "w") as file:
        for line in new_lines:
            file.write(line)

def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


def plot_node(V_est, traj_nodes, leaf_node, graph: graphviz.Digraph, parent_node, index, min_max_stats, parent_name=None):
    def node_name(node, index):
        node_name = f'{index}\n' \
                    f'V = {min_max_stats.normalize(node.mean_value()) if node.num_visits > 0 else 0.0:.3f}\n' \
                    f'N = {node.num_visits}'

        if node == leaf_node:
            node_name += f'\nV_est={V_est:.3f}'
        return node_name

    def edge_label(from_node, to_node):
        action = to_node.action
        return f'a = {to_node.action}\n' \
               f'r = {to_node.reward}\n' \
               f'PUCT = {from_node.puct_scores(min_max_stats)[action]:.3f}\n' \
               f'Q = {from_node.child_values(min_max_stats)[action]:.3f}\n' \
               f'P = {from_node.child_priors[action]:.3f}'

    if parent_name is None:
        parent_name = node_name(parent_node, index)

    if parent_node in traj_nodes:
        graph.node(parent_name, color='red')
    elif parent_node.terminal:
        graph.node(parent_name, color='blue')
    else:
        graph.node(parent_name)
        
    child_names = []
    legal_actions = [12, 19, 29, 30, 40, 52, 86, 97, 100, 106, 107, 108]
    i = 0
    for _, (_, child_node) in enumerate(parent_node.children.items()):
        if child_node.action not in legal_actions:
            continue
        child_name = node_name(child_node, index + i + 1)
        i += 1
        child_names.append(child_name)
        graph.node(child_name)
        graph.edge(parent_name, child_name, label=edge_label(parent_node, child_node))

    if len(parent_node.children.keys()) == 0:
        pass
    else:
        index += len(legal_actions) + 1

    j = 0
    for _, (_, child_node) in enumerate(parent_node.children.items()):
        if child_node.action not in legal_actions:
            continue
        child_name = child_names[j]
        j += 1
        index = plot_node(V_est, traj_nodes, leaf_node, graph, child_node, index, min_max_stats, parent_name=child_name)

    return index


def plot_tree(root_node, leaf_node, V_est, min_max_stats, output_file):
    g = graphviz.Digraph('g', filename='tree.gv', node_attr={'shape': 'circle'})
    nodes = []
    parent = leaf_node
    while parent is not None:
        nodes.append(parent)
        parent = parent.parent_traversed
    plot_node(V_est, nodes, leaf_node, g, root_node, 0, min_max_stats)
    g.save(filename=output_file)
    # g.view()
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MinMaxStats:
    # See: https://arxiv.org/pdf/1911.08265.pdf, p.12
    def __init__(self):
        self.max_delta = 0.01
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        delta = self.maximum - self.minimum
        if delta < self.max_delta:   # See: EfficientZero implementation
            value_norm = (value - self.minimum) / self.max_delta
        else:
            value_norm = (value - self.minimum) / delta
        return value_norm


class DiscreteSupport:
    def __init__(self, min: int, max: int, delta=1.0):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + delta, delta)
        self.size = len(self.range)
        self.delta = delta
