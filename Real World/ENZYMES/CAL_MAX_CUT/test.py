import CAL_MAX_CUT
from CAL_MAX_CUT import load
from CAL_MAX_CUT import graph
from CAL_MAX_CUT import geneticalgorithm
from CAL_MAX_CUT import max_cut
import random
# import pickle


def test(datas):
    datas_total = len(datas)
    best_cut_sum = 0
    s = 0
    for data in datas:
        e = data.edge_index.T
        new_e = e.clone()
        edge = new_e - s - 1
        vertice = data.x.size(0)
        s = s + vertice
        g = graph.Graph(vertice, edge)
        known_cut = [random.choice([0, 1]) for _ in range(vertice)]
        ga = geneticalgorithm.GeneticAlgorithm(g, population_size=100, mutation_rate=0.05, max_generations=100, known_cut=known_cut)
        max_cuts_ga = max_cut.MaxCut(g, ga)
        max_cuts, cuts, isvalid = max_cuts_ga.result()
        best_cut_sum = best_cut_sum + max_cuts
        if isvalid == 0:
            datas_total = datas_total - 1
        print(max_cuts, cuts, isvalid)

    best_cut_average = best_cut_sum / datas_total
    print(best_cut_average)


if __name__ == '__main__':
    paths = ['datasets/test_subgraph_large_enzymes.p', 'datasets/test_subgraph_large_imdb.p', 'datasets/test_subgraph_large_reddit.p']
    load_path = paths[0]
    datas = load.data_load(load_path)
    # f = open('datasets/ENZYMES.p', 'rb')
    # datas = pickle.load(f)
    test(datas)
