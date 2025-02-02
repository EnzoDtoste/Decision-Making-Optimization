from ..problem import Problem
import numpy as np

class MinimumVertexCoverVertices(Problem):
    def __init__(self, choice, choiceParameters):
        super().__init__(choice, choiceParameters)

    def get_current_embedding(self, params):
        map, _ = params
        return np.array(map)

    def get_choices(self, params):
        _, nodes = params
        return nodes

    def order_choices(self, choices, params):
        map, _ = params
        return sorted(choices, key=lambda node: sum(map[node]), reverse=True)

    def run(self, graph):
        self.reset_embeddings()
        self.choiceParameters.reset_state()

        n = len(graph)
        nodes = [i for i in range(n)]
        dominating_set = set()
        dominated_mask = [False] * n
        map = [[1 if col in graph[row] or col == row else 0 for col in nodes] for row in nodes]

        while not all(dominated_mask):
            node = self.select_choice([map, nodes])
            nodes.remove(node)
            dominating_set.add(node)
            map[node][node] = 0
            dominated_mask[node] = True
            for neighbor in graph[node]:
                if not dominated_mask[neighbor]:
                    for v in graph[neighbor]:
                        map[v][neighbor] = 0
                    dominated_mask[neighbor] = True
                    map[neighbor][neighbor] = 0
                map[neighbor][node] = 0
        return len(dominating_set)