import CAL_MAX_CUT
from CAL_MAX_CUT import chromosome
import random


class GeneticAlgorithm:
    def __init__(self, graph, population_size, mutation_rate, max_generations, known_cut=None):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = []

        # 初始化种群
        if known_cut:
            self.population.append(chromosome.Chromosome(known_cut))  # 加入已知的割
            population_size -= 1  # 由于已经添加了一个已知的割，所以种群大小减1
        for _ in range(population_size):
            genes = [random.choice([0, 1]) for _ in range(graph.vertices)]
            self.population.append(chromosome.Chromosome(genes))

    def run(self):
        best_chromosome = None
        # 这里需要先计算初始种群的适应度
        for individual in self.population:
            individual.evaluate_fitness(self.graph)
        # 确定种群中当前最优的染色体
        best_chromosome = max(self.population, key=lambda x: x.fitness)

        for generation in range(self.max_generations):
            new_population = []
            # 生成后代直到达到种群大小
            while len(new_population) < self.population_size:
                # 轮盘选择
                parent1, parent2 = self.select_parents()

                # 进行交叉和变异
                offspring = parent1.crossover(parent2)
                offspring.mutate(self.mutation_rate)
                offspring.evaluate_fitness(self.graph)  # 计算新个体的适应度

                # 将新个体添加到新种群中
                new_population.append(offspring)

            # 更新当前种群
            self.population = new_population
            # 确定种群中当前最优的染色体
            current_best = max(self.population, key=lambda x: x.fitness)
            # 如果当前最优的染色体比目前为止找到的最优染色体更优，则更新最优染色体
            if current_best.fitness > best_chromosome.fitness:
                best_chromosome = current_best

        # 返回目前为止找到的最优染色体
        return best_chromosome

    def linear_ranking_selection(self, pmin, pmax):
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        rank_probabilities = [pmin + (pmax - pmin) * i / (self.population_size-1) for i in range(self.population_size)]
        selected_index = random.choices(range(self.population_size), weights=rank_probabilities, k=1)[0]
        return sorted_population[selected_index]

    def select_parents(self):
        parent1 = self.linear_ranking_selection(pmin=0.1, pmax=0.9)
        parent2 = self.linear_ranking_selection(pmin=0.1, pmax=0.9)
        return parent1, parent2
