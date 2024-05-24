import os

import click
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt

from GraphData import NodeLabeling
from GraphData.DataSplits.load_splits import Load_Splits
from GraphData.GraphData import GraphData, get_graph_data
from GraphData.Labels.generator.load_labels import load_labels
from Layers.GraphLayers import Layer
from NeuralNetArchitectures import GraphNN
from TrainTestData import TrainTestData as ttd
from utils.Parameters.Parameters import Parameters
from utils.RunConfiguration import RunConfiguration


def draw_graph(graph_data: GraphData, graph_id, ax, node_size=50, edge_color='black',
               edge_width=0.5, draw_type='circle'):
    graph = graph_data.graphs[graph_id]

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for node in graph.nodes():
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break

    node_labels = {}
    for node in graph.nodes():
        key = int(node)
        value = str(graph_data.node_labels['primary'].node_labels[graph_id][node])
        node_labels[key] = f'{value}'
    edge_labels = {}
    for (key1, key2, value) in graph.edges(data=True):
        if "label" in value and len(value["label"]) > 0:
            edge_labels[(key1, key2)] = int(value["label"])
        else:
            edge_labels[(key1, key2)] = ""
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='black')
    # get node colors from the node labels using the plasma colormap
    cmap = plt.get_cmap('tab20')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=graph_data.node_labels['primary'].num_unique_node_labels)
    node_colors = [cmap(norm(graph_data.node_labels['primary'].node_labels[graph_id][node])) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)


def draw_resize_layer(results_path: str, graph_data: GraphData, graph_id, layer, ax, cmap='seismic', node_size=50,
                      edge_width=5, draw_type='circle'):
    weight = layer.get_weights()
    bias = layer.get_bias()
    layer_id = layer.layer_id
    # get the adjacency matrices + the bias vector for the first graph
    # load from txt file
    path = f'{results_path}{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt'
    # read as csv with pandas
    df = pd.read_csv(path, sep=';', header=None)

    graph = graph_data.graphs[graph_id]
    weight_matrix = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    # iterate over the rows and columns of the weight matrix
    for i in range(0, graph.number_of_nodes()):
        for j in range(0, graph.number_of_nodes()):
            if df.iloc[i, j] != 0:
                weight_matrix[i][j] = weight[df.iloc[i, j] - 1]
    bias_vector = np.asarray(bias)

    weight_min = np.min(weight_matrix)
    weight_max = np.max(weight_matrix)
    bias_min = np.min(bias_vector)
    bias_max = np.max(bias_vector)

    weight_max_abs = max(abs(weight_min), abs(weight_max))
    bias_max_abs = max(abs(bias_min), abs(bias_max))

    # use seismic colormap with maximum and minimum values from the weight matrix
    cmap = plt.get_cmap(cmap)
    # normalize item number values to colormap
    norm_weight = matplotlib.colors.Normalize(vmin=-weight_max_abs, vmax=weight_max_abs)
    norm_bias = matplotlib.colors.Normalize(vmin=-bias_max_abs, vmax=bias_max_abs)
    weight_colors = cmap(norm_weight(weight_matrix))
    bias_colors = cmap(norm_bias(bias_vector))

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for i, node in enumerate(graph.nodes()):
        if i == 0:
            print(f"First node: {graph_data.node_labels['primary'].node_labels[graph_id][node]}")
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    # graph to digraph with
    digraph = nx.DiGraph()
    # get indices of the 100 largest weights (absolute value)
    #highest_weight_number = 100
    #weight_indices = np.unravel_index(np.argsort(np.abs(weight_matrix), axis=None), weight_matrix.shape)
    #weight_indices = (weight_indices[0][-highest_weight_number:], weight_indices[1][-highest_weight_number:])
    # set all other weights in the weight matrix to zero
    #non_zero_weights = np.zeros_like(weight_matrix)
    #non_zero_weights[weight_indices] = weight_matrix[weight_indices]
    #weight_matrix = non_zero_weights
    # add all edges where weight is not zero
    for i in range(0, graph.number_of_nodes()):
        for j in range(0, graph.number_of_nodes()):
            if weight_matrix[i][j] != 0:
                digraph.add_edge(i, j)
                digraph.add_edge(j, i)
    curved_edges = [edge for edge in digraph.edges() if reversed(edge) in digraph.edges()]
    curved_edges_colors = []
    edge_widths = []
    for edge in curved_edges:
        curved_edges_colors.append(weight_colors[edge[0]][edge[1]])
        edge_widths.append(edge_width * abs(weight_matrix[edge[0]][edge[1]]) / weight_max_abs)
    arc_rad = 0.25
    nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                           width=edge_widths,
                           connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=5)

    node_colors = []
    node_sizes = []
    for node in digraph.nodes():
        node_label = graph_data.node_labels['primary'].node_labels[graph_id][node]
        node_colors.append(bias_colors[node_label])
        node_sizes.append(node_size * abs(bias_vector[node_label]) / bias_max_abs)

    nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)


def draw_graph_layer(results_path: str, graph_data: GraphData, graph_id, layer, ax, cmap='seismic', node_size=50,
                     edge_width=5, draw_type='circle', filter_weights=True, percentage=0.1, absolute=None):
    all_weights = layer.get_weights()
    bias = layer.get_bias()
    graph = graph_data.graphs[graph_id]
    weight_distribution = layer.weight_distribution[graph_id]
    graph_weights = np.zeros_like(all_weights)
    for entry in weight_distribution:
        graph_weights[entry[2]] = all_weights[entry[2]]
    graph_weights = np.asarray(graph_weights)



    # sort weights
    if filter_weights:
        sorted_weights = np.sort(graph_weights)
        if absolute is None:
            lower_bound_weight = sorted_weights[int(len(sorted_weights) * percentage) - 1]
            upper_bound_weight = sorted_weights[int(len(sorted_weights) * (1 - percentage))]
        else:
            lower_bound_weight = sorted_weights[absolute - 1]
            upper_bound_weight = sorted_weights[-absolute]
        # set all weights smaller than the lower bound and larger than the upper bound to zero
        upper_weights = np.where(graph_weights >= upper_bound_weight, graph_weights, 0)
        lower_weights = np.where(graph_weights <= lower_bound_weight, graph_weights, 0)

        weights = upper_weights + lower_weights
    else:
        weights = np.asarray(graph_weights)
    bias_vector = np.asarray(bias)

    weight_min = np.min(graph_weights)
    weight_max = np.max(graph_weights)
    weight_max_abs = max(abs(weight_min), abs(weight_max))
    bias_min = np.min(bias_vector)
    bias_max = np.max(bias_vector)
    bias_max_abs = max(abs(bias_min), abs(bias_max))

    # use seismic colormap with maximum and minimum values from the weight matrix
    cmap = plt.get_cmap(cmap)
    # normalize item number values to colormap
    norm_weight = matplotlib.colors.Normalize(vmin=weight_min, vmax=weight_max)
    norm_bias = matplotlib.colors.Normalize(vmin=bias_min, vmax=bias_max)
    weight_colors = cmap(norm_weight(graph_weights))
    bias_colors = cmap(norm_bias(bias_vector))

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for i, node in enumerate(graph.nodes()):
        if i == 0:
            print(f"First node: {graph_data.node_labels['primary'].node_labels[graph_id][node]}")
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    # graph to digraph with
    digraph = nx.DiGraph()
    for node in graph.nodes():
        digraph.add_node(node)

    edge_widths = []
    for entry in weight_distribution:
        i = entry[0]
        j = entry[1]
        if weights[entry[2]] != 0:
            # add edge with weight as data
            digraph.add_edge(i, j, weight=entry[2])
    curved_edges = [edge for edge in digraph.edges(data=True)]
    curved_edges_colors = []

    for edge in curved_edges:
        curved_edges_colors.append(weight_colors[edge[2]['weight']])
        edge_widths.append(edge_width * abs(weights[edge[2]['weight']]) / weight_max_abs)
    arc_rad = 0.25
    nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                           width=edge_widths,
                           connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=5)

    node_colors = []
    node_sizes = []
    for node in digraph.nodes():
        node_label = graph_data.node_labels['primary'].node_labels[graph_id][node]
        node_colors.append(bias_colors[node_label])
        node_sizes.append(node_size * abs(bias_vector[node_label]) / bias_max_abs)

    nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)


@click.command()
@click.option('--data_path', default="../GraphBenchmarks/Data/", help='Path to the graph data')
@click.option('--db', default="EvenOddRings2_16", help='Database to use')
@click.option('--config', default="", help='Path to the configuration file')
@click.option('--out', default="")
@click.option('--draw_type', default='circle')
# --data_path ../GraphBenchmarks/Data/ --db EvenOddRings2_16 --config ../TEMP/EvenOddRings2_16/config.yml
def main(data_path, db, config, out, draw_type):
    run = 0
    k_val = 0
    kFold = 10
    # load the model
    configs = yaml.safe_load(open(config))
    # get the data path from the config file
    data_path = f"../{configs['paths']['data']}"
    r_path = f"../{configs['paths']['results']}"
    distance_path = f"../{configs['paths']['distances']}"
    splits_path = f"../{configs['paths']['splits']}"
    results_path = r_path + db + "/Results/"

    graph_data = get_graph_data(data_path=data_path, db_name=db, distance_path=distance_path,
                                use_features=configs['use_features'], use_attributes=configs['use_attributes'])
    # adapt the precision of the input data
    if 'precision' in configs:
        if configs['precision'] == 'double':
            for i in range(len(graph_data.inputs)):
                graph_data.inputs[i] = graph_data.inputs[i].double()

    #create run config from first config
    # get all different run configurations
    # define the network type from the config file
    run_configs = []
    # iterate over all network architectures
    for network_architecture in configs['networks']:
        layers = []
        # get all different run configurations
        for l in network_architecture:
            layers.append(Layer(l))
        for b in configs['batch_size']:
            for lr in configs['learning_rate']:
                for e in configs['epochs']:
                    for d in configs['dropout']:
                        for o in configs['optimizer']:
                            for loss in configs['loss']:
                                run_configs.append(RunConfiguration(network_architecture, layers, b, lr, e, d, o, loss))
    for i, run_config in enumerate(run_configs):
        config_id = str(i).zfill(6)
        model_path = f'{r_path}{db}/Models/model_Configuration_{config_id}_run_{run}_val_step_{k_val}.pt'
        seed = k_val + kFold * run
        data = Load_Splits(f"../{configs['paths']['splits']}", db)
        test_data = np.asarray(data[0][k_val], dtype=int)
        training_data = np.asarray(data[1][k_val], dtype=int)
        validate_data = np.asarray(data[2][k_val], dtype=int)
        # check if the model exists
        try:
            with open(model_path, 'r'):
                """
                Set up the network and the parameters
                """

                para = Parameters()
                """
                    Data parameters
                """
                para.set_data_param(path=data_path, results_path=results_path,
                                    splits_path=splits_path,
                                    db=db,
                                    max_coding=1,
                                    layers=run_config.layers,
                                    batch_size=run_config.batch_size, node_features=1,
                                    load_splits=configs['load_splits'],
                                    configs=configs,
                                    run_config=run_config, )

                """
                    Network parameters
                """
                para.set_evaluation_param(run_id=run, n_val_runs=kFold, validation_id=k_val,
                                          config_id=config_id,
                                          n_epochs=run_config.epochs,
                                          learning_rate=run_config.lr, dropout=run_config.dropout,
                                          balance_data=configs['balance_training'],
                                          convolution_grad=True,
                                          resize_graph=True)

                """
                Print, save and draw parameters
                """
                para.set_print_param(no_print=False, print_results=False, net_print_weights=True,
                                     print_number=1,
                                     draw=False, save_weights=False,
                                     save_prediction_values=False, plot_graphs=False,
                                     print_layer_init=False)

                for l in run_config.layers:
                    label_path = f"../GraphData/Labels/{db}_{l.get_layer_string()}_labels.txt"
                    if os.path.exists(label_path):
                        g_labels = load_labels(path=label_path)
                        graph_data.node_labels[l.get_layer_string()] = g_labels
                    else:
                        # raise an error if the file does not exist
                        raise FileNotFoundError(f"File {label_path} does not exist")

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)

                net = GraphNN.GraphNet(graph_data=graph_data,
                                       para=para,
                                       seed=seed)

                net.load_state_dict(torch.load(model_path))
                # evaluate the performance of the model on the test data
                outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
                with torch.no_grad():
                    for j, data_pos in enumerate(test_data, 0):
                        inputs = torch.DoubleTensor(graph_data.inputs[data_pos])
                        outputs[j] = net(inputs, data_pos)
                    labels = graph_data.one_hot_labels[test_data]
                    # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                    counter = 0
                    correct = 0
                    for i, x in enumerate(outputs, 0):
                        if torch.argmax(x) == torch.argmax(labels[i]):
                            correct += 1
                        counter += 1
                    accuracy = correct / counter
                    print(f"Accuracy for model {model_path} is {accuracy}")
                # get the first three graphs from the test data
                graph_ids = test_data[[0, 20, 40]]
                rows = len(graph_ids)
                cols = len(net.net_layers)
                fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))
                plt.subplots_adjust(wspace=0, hspace=0)
                layers = net.net_layers
                # run over axes array
                for i, x in enumerate(axes):
                    graph_id = graph_ids[i]
                    for j, ax in enumerate(x):
                        if j == 0:
                            draw_graph(graph_data, graph_id, ax, node_size=40, edge_width=2, draw_type=draw_type)
                            # set title on the left side of the plot
                            ax.set_ylabel(f"Graph Label: {graph_data.graph_labels[graph_id]}")
                        else:
                            if i == 0:
                                ax.set_title(f"Convolution Layer {j}")
                            draw_graph_layer(results_path=r_path, graph_data=graph_data, graph_id=graph_id,
                                             layer=layers[j - 1], ax=ax, node_size=40, edge_width=2,
                                             draw_type=draw_type, filter_weights=True, percentage=0.01, absolute=None)

                # draw_graph_layer(graph_data, graph_id, net.lr)
                # save the figure as svg
                plt.savefig(f'{out}/{db}_weights_run_{run}_val_step_{k_val}.svg')
                plt.show()






        except FileNotFoundError:
            print(f"Model {model_path} not found")
            return


if __name__ == "__main__":
    main()
