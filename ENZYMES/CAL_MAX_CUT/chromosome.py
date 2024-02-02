import CAL_MAX_CUT
import random
import torch


class Chromosome:
    def __init__(self, genes):
        self.genes = torch.as_tensor(genes, dtype=torch.int)
        self.fitness = -1

    def evaluate_fitness(self, graph):
        cut_weight = 0
        if graph.is_connected_1(self.genes) and graph.is_connected_0(self.genes):
            for edge in graph.edges:
                u, v = edge
                if self.genes[u] != self.genes[v] and u > v:  # 如果边的两个顶点在割的不同侧，则是割边
                    cut_weight += 1
            self.fitness = cut_weight  # 适应度是所有割边的和
        else:
            # 如果子图不连通，可以赋予极低的适应度
            self.fitness = 0
        return self.fitness

    def crossover(self, other):
        point = random.randint(0, len(self.genes))
        return Chromosome(torch.cat((self.genes[:point], other.genes[point:]), dim=0))

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]
