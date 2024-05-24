import pickle
import time

import networkx as nx

from GraphData.GraphData import get_graph_data


def write_distances(graph_data, db_name, cutoff, distance_path="") -> int:
    # get current time
    start = time.time()
    max_distance = 0
    distances = []
    for graph in graph_data.graphs:
        d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
        # order the dictionary by the values
        for _, value in d.items():
            for _, distance in value.items():
                if distance > max_distance:
                    max_distance = distance
        distances.append(d)
    # save list of dictionaries to a pickle file
    pickle.dump(distances, open(f"{distance_path}{db_name}_distances.pkl", 'wb'))
    print(f"Max distance for {db_name}: {max_distance}")
    # get the runtime
    end = time.time()
    print(f"Runtime: {end - start}")
    return max_distance


def save_distances(data_path="../../../GraphData/DS_all/", db_names=[], cutoff=None, distance_path=""):
    for db_name in db_names:
        graph_data = get_graph_data(db_name=db_name, data_path=data_path, with_distances=False)
        write_distances(graph_data=graph_data, db_name=db_name, cutoff=cutoff, distance_path=distance_path)


def main():
    save_distances(data_path="../TUGraphs/", db_names=['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI'],
                   distance_path="")
    save_distances(data_path="../../GraphBenchmarks/Data/",
                   db_names=['LongRings100', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'CSL', 'Snowflakes'],
                   distance_path="")


if __name__ == '__main__':
    main()
