from typing import List

import networkx as nx


def standard_edge_labeling(graphs:List[nx.Graph]):
    """
    Standard edge labeling method. It gets the edge labels from the graphs in graph_data
    :param graph_data: GraphData object
    :param edge_labels: List of edge labels
    :param unique_edge_labels: Dictionary of unique edge labels per graph
    :param db_unique_edge_labels: Dictionary of unique edge labels per database
    :return: None
    """
    edge_labels = []
    unique_edge_labels = []
    db_unique_edge_labels = {}
    for graph in graphs:
        edge_labels.append([])
        unique_edge_labels.append({})
        for edge in graph.edges(data=True):
            if 'label' in edge[2]:
                if type(edge[2]['label']) == int:
                    edge_label = edge[2]['label']
                elif type(edge[2]['label']) == list and len(edge[2]['label']) > 0:
                    edge_label = edge[2]['label'][0]
                else:
                    edge_label = 0
                edge_labels[-1].append(edge_label)
                if edge_label not in unique_edge_labels[-1]:
                    unique_edge_labels[-1][edge_label] = 1
                else:
                    unique_edge_labels[-1][edge_label] += 1
                if edge_label not in db_unique_edge_labels:
                    db_unique_edge_labels[edge_label] = 1
                else:
                    db_unique_edge_labels[edge_label] += 1
            else:
                edge_labels[-1].append(0)
                if 0 not in unique_edge_labels[-1]:
                    unique_edge_labels[-1][0] = 1
                else:
                    unique_edge_labels[-1][0] += 1
                if 0 not in db_unique_edge_labels:
                    db_unique_edge_labels[0] = 1
                else:
                    db_unique_edge_labels[0] += 1
    return edge_labels, unique_edge_labels, db_unique_edge_labels