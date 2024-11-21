import numpy as np
from mealpy import PermutationVar, GWO, SCA, WOA, FOX, Problem
import matplotlib.pyplot as plt

# Posiciones de las ciudades
city_positions = np.array([[60, 200], [180, 200], [80, 180], [140, 180], [20, 160],
                           [100, 160], [200, 160], [140, 140], [40, 120], [100, 120],
                           [180, 100], [60, 80], [120, 80], [180, 60], [20, 40],
                           [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]])
num_cities = len(city_positions)

# Datos del problema
data = {
    "city_positions": city_positions,
    "num_cities": num_cities,
}

# Clase personalizada para TSP
class TspProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    @staticmethod
    def calculate_distance(city_a, city_b):
        # Distancia Euclidiana entre dos ciudades
        return np.linalg.norm(city_a - city_b)

    @staticmethod
    def calculate_total_distance(route, city_positions):
        
        total_distance = 0
        num_cities = len(route)
        for idx in range(num_cities):
            current_city = route[idx]
            next_city = route[(idx + 1) % num_cities]  # Vuelve al inicio
            total_distance += TspProblem.calculate_distance(city_positions[current_city], city_positions[next_city])
        return total_distance

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        route = x_decoded["per_var"]
        fitness = self.calculate_total_distance(route, self.data["city_positions"])
        return fitness

# Límite para las permutaciones
bounds = PermutationVar(valid_set=list(range(0, num_cities)), name="per_var")


problem = TspProblem(bounds=bounds, minmax="min", data=data)

# Lista de metaheurísticas
metaheuristicas = [
    GWO.OriginalGWO(epoch=100, pop_size=20),
    SCA.OriginalSCA(epoch=100, pop_size=20),
    WOA.OriginalWOA(epoch=100, pop_size=20),
    FOX.OriginalFOX(epoch=100, pop_size=20),
]

# Función para graficar la ruta
def plot_route(route, city_positions, title):

    closed_route = route + [route[0]]
    coords = city_positions[closed_route]  

    plt.figure(figsize=(10, 8))
    plt.plot(coords[:, 0], coords[:, 1], marker='o')
    for i, (x, y) in enumerate(coords[:-1]):  
        plt.text(x, y, str(route[i]), fontsize=12)  
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def plot_convergence(histories, labels):
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    plt.title("Convergencia de fitness")
    plt.xlabel("Épocas")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejecución de metaheurísticas y recolección de datos
all_histories = []
all_labels = []
fitness_list = [] 

for mh in metaheuristicas:
   
    mh.solve(problem)
    best_route = mh.problem.decode_solution(mh.g_best.solution)["per_var"]
    best_fitness = mh.g_best.target.fitness

    # Recolección de historial de fitness 
    fitness_history = [agent.target.fitness for agent in mh.history.list_global_best]
    all_histories.append(fitness_history)
    all_labels.append(mh.name)

 
    for i, fit in enumerate(fitness_history):
        fitness_list.append((mh.name, i + 1, fit))


    print(f"Metaheurística: {mh.name}")
    print(f"Mejor ruta: {best_route}")
    print(f"Distancia total: {best_fitness}\n")


    plot_route(best_route, city_positions, f"Ruta óptima - {mh.name}")


plot_convergence(all_histories, all_labels)


print("\nLista de fitness por metaheurística y número de iteración:")
for entry in fitness_list:
    print(f"Método: {entry[0]}, Iteración: {entry[1]}, Fitness: {entry[2]}")
