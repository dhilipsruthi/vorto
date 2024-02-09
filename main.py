
# In the pursuit of devising an optimal solution for a complex optimization problem, I have amalgamated the renowned
# Clarke-Wright algorithm with the sophisticated technique of Guided Ejection Search.
# This strategic combination leverages the strength of the Clarke-Wright algorithm to establish seed data,
# providing a foundational basis for subsequent optimization endeavors. Building upon this groundwork,
# the application of the Guided Ejection Search method serves to judiciously modify the existing dataset,
# culminating in an intricately refined solution.

# This innovative approach is inspired by the scholarly work titled "Large Neighbourhood Search with Adaptive
# Guided Ejection Search for the Pickup and Delivery Problem with Time Windows."


# importing necessary libraries
import sys
import math
import time
import copy
import os
import threading
import random

# setting the max miles per truck
MAX_MILES_PER_TRUCK = 720.0

# declaring necessary variables
Route = list[int]
Plan = list[Route]
data = None
distance_matrix = None
distance_saved = None
thread_solutions = None
thread_solution_costs = None

# formating the input data to usable format
class Package:
    def __init__(self, id, start_x, start_y, drop_x, drop_y, length) -> None:
        self.id: int = id
        self.start_x: float = start_x
        self.start_y: float = start_y
        self.drop_x: float = drop_x
        self.drop_y: float = drop_y
        self.length: float = length

# function to find distance between 2 points
def get_distance(point1, point2):
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2.0) + math.pow(point1[1] - point2[1], 2.0))
    return distance

# function to get x,y coordiantes from input
def get_coordinates(coord_str: str):
    # Remove parentheses and split by comma
    numbers = coord_str[1:-1].split(',')
    # Convert to float
    x = float(numbers[0])
    y = float(numbers[1])
    return x, y

# read input file and generate distance and distance saved matrix
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
        # holds the distance between any 2 load points
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
                    dis = ((distance_matrix[0][i] + distance_matrix[i][0] + distance_matrix[0][j] +
                                  distance_matrix[j][0])) - \
                                (distance_matrix[0][i] + distance_matrix[i][j] + distance_matrix[j][0])
                    distance_saved.append([dis, i, j, ])
        # holds the distance saved by traveling from one load to other
        distance_saved.sort(reverse=True)
    return data

# initialise a path that is feasible
def get_seed_route(problem_def: list[Package]) -> list[Route]:
    visit = []
    route_list: list[Route] = []
    i = random.randint(0,len(distance_saved)-1)
    #check if the distance between travling from origin, one load to another back to origin is less than visiting origin
    #evey single time
    for j in range(i,len(distance_saved)):
        diff = distance_saved[j][0]
        i1 = distance_saved[j][1]
        i2 = distance_saved[j][2]
        dis = (distance_matrix[0][i1] + distance_matrix[i1][0] + distance_matrix[0][i2] + distance_matrix[i2][0]) - diff
        #if less add that to the route
        if dis <= MAX_MILES_PER_TRUCK and i1 not in visit and i2 not in visit:
            visit.append(i1)
            visit.append(i2)
            route_list.append([i1,i2])
    if len(visit) != len(data)-1:
        for k in range(1,len(data)):
            if k not in visit:
                visit.append(k)
                route_list.append([k])

    return route_list

# function to get total cost
def cost(plan: Plan) -> float:
    total_cost = 500 * len(plan)

    for route in plan:
        route_cost = get_route_cost(route)

        total_cost += route_cost

    return total_cost

# function to get current truck travel time
def get_route_cost(route: Route) -> float:
    route_cost = distance_matrix[0][route[0]] + distance_matrix[route[-1]][0]

    for pkg_idx in range(1, len(route)):
        route_cost += distance_matrix[route[pkg_idx - 1]][route[pkg_idx]]

    return route_cost

# get the shortest route
def get_shortest_route(plan: list[Route]) -> int:
    shortest_idx = random.randint(0, len(plan) - 1)
    shorest_so_far = len(plan[shortest_idx])

    for i in range(0, len(plan)):
        if len(plan[i]) < shorest_so_far:
            shorest_so_far = len(plan[i])
            shortest_idx = i

    return shortest_idx

# check if the current route is viable
def is_feasible(plans: list[Route]):
    i = 0
    for route in plans:
        route_cost: float = distance_matrix[0][route[0]] + distance_matrix[route[-1]][0]

        for pkg_idx in range(1, len(route)):
            route_cost += distance_matrix[route[pkg_idx - 1]][route[pkg_idx]]

        if (route_cost > MAX_MILES_PER_TRUCK):
            return (False, i)

        i += 1
    return (True, -1)

#perform the main logic
def perform_optimization(thread_idx: int):
    time.sleep(.1)
    global thread_solutions
    global thread_solution_costs
    global data

    # Create a seed list of routes that are feasible
    seed_route: Route = get_seed_route(data)

    best_plan = seed_route
    best_cost = cost(best_plan)

    start_time = time.time_ns()

    current_plan: list[Route] = copy.deepcopy(best_plan)
    dropped_packages: list[int] = []

    #iterate till time reached
    num_iterations_since_improvement = 0
    while (time.time_ns() - start_time) < 25 * 1e9:
        factor = random.uniform(0, 1)

        if (len(dropped_packages) > 0):  # Need to assign package somewhere
            # find truck having highest time left
            slacker_truck = get_shortest_route(current_plan)
            random_pickup = random.randint(0, len(dropped_packages) - 1)
            current_plan[slacker_truck].append(dropped_packages.pop(random_pickup))
        else:  #destroy route to check for better solution
            if (factor <= 0.1 and len(current_plan) > 1):
                # remove a truck
                truck_idx = random.randint(0, len(current_plan) - 1)
                for package in current_plan[truck_idx]:
                    dropped_packages.append(package)
                current_plan.pop(truck_idx)
            elif (factor <= 0.4):
                # random shuffle within truck, switch 2
                truck = random.randint(0, len(current_plan) - 1)
                pkg1 = random.randint(0, len(current_plan[truck]) - 1)
                pkg2 = random.randint(0, len(current_plan[truck]) - 1)
                current_plan[truck][pkg1], current_plan[truck][pkg2] = current_plan[truck][pkg2], current_plan[truck][
                    pkg1]
            elif (factor <= 0.6 and len(current_plan) > 1):
                # trade with truck via teleport
                truck1 = random.randint(0, len(current_plan) - 1)
                pkg1 = random.randint(0, len(current_plan[truck1]) - 1)

                truck2 = random.randint(0, len(current_plan) - 1)
                pkg2 = random.randint(0, len(current_plan[truck2]) - 1)

                current_plan[truck1][pkg1], current_plan[truck2][pkg2] = current_plan[truck2][pkg2], \
                                                                         current_plan[truck1][pkg1]
            elif (factor <= 0.8 and len(current_plan) > 1):
                truck1 = random.randint(0, len(current_plan) - 1)
                truck2 = random.randint(0, len(current_plan) - 1)

                while (truck2 == truck1):
                    truck2 = random.randint(0, len(current_plan) - 1)

                for i in range(len(current_plan[truck2])):
                    current_plan[truck1].append(current_plan[truck2].pop(0))

                current_plan.pop(truck2)
            else:
                # drop current route
                truck = random.randint(0, len(current_plan) - 1)
                pkg = random.randint(0, len(current_plan[truck]) - 1)

                dropped_packages.append(current_plan[truck].pop(pkg))
                if (len(current_plan[truck]) == 0):
                    current_plan.pop(truck)

        truck_feas, bad_truck = is_feasible(current_plan)
        if (not truck_feas):
            # break the bad route
            split_idx = random.randint(1, len(current_plan[bad_truck]) - 1)
            new_truck: list[int] = []

            for i in range(split_idx, len(current_plan[bad_truck])):
                new_truck.append(current_plan[bad_truck].pop(split_idx))

            current_plan.append(copy.deepcopy(new_truck))

        truck_feas, bad_truck = is_feasible(current_plan)
        feasible = truck_feas and len(dropped_packages) == 0
        #continue till we find a feasible solution
        if (not feasible):
            continue
        current_cost = cost(current_plan)
        #if we found a better solution update best solution
        if current_cost < best_cost:
            num_iterations_since_improvement = 0
            best_cost = current_cost
            best_plan = copy.deepcopy(current_plan)
        #selecting a point to move away from local best solution
        if (num_iterations_since_improvement > 3 * len(data)):
            current_plan = get_seed_route(data)
            num_iterations_since_improvement = 0
        else:
            num_iterations_since_improvement += 1
    #save results to thread
    thread_solutions[thread_idx] = best_plan
    thread_solution_costs[thread_idx] = best_cost


if __name__ == "__main__":
    random.seed(432638267)
    filename = sys.argv[1]
    data: list[Package] = parse_file(filename)
    #get total number of cpu avaliable
    num_threads = os.cpu_count()

    thread_solutions = [-1] * num_threads
    thread_solution_costs = [float("inf")] * num_threads
    #form threads
    threads = [-1] * num_threads
    for i in range(num_threads):
        threads[i] = threading.Thread(target=perform_optimization, args=[i])
    #start thread
    for i in range(num_threads):
        threads[i].start()
    #wait till all threads finished
    for i in range(num_threads):
        threads[i].join()

    overal_best = thread_solutions[0]
    overal_best_cost = thread_solution_costs[0]
    #find the best performing thread
    for i in range(1, num_threads):
        if (thread_solution_costs[i] < overal_best_cost):
            overal_best_cost = thread_solution_costs[i]
            overal_best = thread_solutions[i]
    #print the output
    for truck_route in overal_best:
        print(truck_route)
