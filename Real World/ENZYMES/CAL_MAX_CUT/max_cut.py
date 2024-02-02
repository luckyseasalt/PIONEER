import CAL_MAX_CUT


class MaxCut:
    def __init__(self, graph, ga):
        self.graph = graph
        self.ga = ga

    def result(self):
        best_cut = self.ga.run()
        if best_cut.fitness != 0:
            flag = 1
            return best_cut.fitness, best_cut.genes, flag
        else:
            flag = 0
            return best_cut.fitness, None, flag
