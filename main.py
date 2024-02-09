import sys
import math
import time
import copy
import os
import threading
import random

MAX_MILES_PER_CAR = 720.0

Route = list[int]
Plan = list[Route]

data = None
distance_matrix = None
distance_saved = None
thread_solutions = None
thread_solution_costs = None
class Package:
    def __init__(self, id, start_x, start_y, drop_x, drop_y, length) -> None:
        self.id: int = id
        self.start_x: float = start_x
        self.start_y: float = start_y
        self.drop_x: float = drop_x
        self.drop_y: float = drop_y
        self.length: float = length
def get_distance(point1, point2):
    # Calculate the sum of squared differences for each dimension
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2.0) + math.pow(point1[1] - point2[1], 2.0))

    return distance
def get_coordinates(coord_str: str):
    # Remove parentheses and split by comma
    numbers = coord_str[1:-1].split(',')

    # Convert to float
    x = float(numbers[0])
    y = float(numbers[1])

    return x, y
def parse_file(file_path: str) -> list[Package]:
    global distance_matrix
    global distance_saved
    data = []
    first = True

    with open(file_path, 'r') as file:
        # Read and print each line
        for line in file:
            new_line = line.split()

            if first:
                first = False
                data.append(Package(0, 0.0, 0.0, 0.0, 0.0, 0.0))
            else:
                x1, y1 = get_coordinates(new_line[1])
                x2, y2 = get_coordinates(new_line[2])
                length = get_distance((x1, y1), (x2, y2))

                package = Package(int(new_line[0]), float(x1), float(y1), float(x2), float(y2), float(length))
                data.append(package)

        distance_matrix = [[0.0 for _ in range(len(data))] for _ in range(len(data))]

        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                fro: Package = data[i]
                to: Package = data[j]

                distance_matrix[i][j] = get_distance((fro.drop_x, fro.drop_y),
                                                           (to.start_x, to.start_y)) + fro.length
                distance_matrix[j][i] = get_distance((to.drop_x, to.drop_y),
                                                           (fro.start_x, fro.start_y)) + to.length
        distance_saved = []
        for i in range(1, len(data)):
            for j in range(1, len(data)):
                if i == j:
                    continue
                else:
                    dis_saved = ((distance_matrix[0][i] + distance_matrix[i][0] + distance_matrix[0][j] +
                                  distance_matrix[j][0])) - \
                                (distance_matrix[0][i] + distance_matrix[i][j] + distance_matrix[j][0])
                    distance_saved.append([dis_saved, i, j, ])
        distance_saved.sort(reverse=True)

    return data

def get_seed_route(problem_def: list[Package]) -> list[Route]:
    visit = []
    route_list: list[Route] = []
    i = random.randint(0,len(distance_saved)-1)
    for j in range(i,len(distance_saved)):
        diff = distance_saved[j][0]
        i1 = distance_saved[j][1]
        i2 = distance_saved[j][2]
        dis = (distance_matrix[0][i1] + distance_matrix[i1][0] + distance_matrix[0][i2] + distance_matrix[i2][0]) - diff
        if dis <= MAX_MILES_PER_CAR and i1 not in visit and i2 not in visit:
            visit.append(i1)
            visit.append(i2)
            route_list.append([i1,i2])
    if len(visit) != len(data)-1:
        for k in range(1,len(data)):
            if k not in visit:
                visit.append(k)
                route_list.append([k])
    return route_list
if __name__ == "__main__":
    random.seed(432638267)
    filename = sys.argv[1]
    data: list[Package] = parse_file(filename)
