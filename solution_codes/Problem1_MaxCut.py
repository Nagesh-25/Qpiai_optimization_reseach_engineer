# \"\"\"Problem2_VRPTW.py
# Detailed skeleton for Vehicle Routing Problem with Time Windows (VRPTW) using OR-Tools.
# Produces:
#  - Synthetic dataset generator (CSV)
#  - OR-Tools example to solve VRPTW (if ortools installed)
#  - Route visualization using matplotlib
# Usage:
#     python Problem2_VRPTW.py
# Notes:
#  - OR-Tools installation: pip install ortools
#  - If ortools is not available, the script will save the synthetic dataset and provide instructions.
# \"\"\"

import os, csv, math, random, time
import numpy as np
import matplotlib.pyplot as plt

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except Exception as e:
    ORTOOLS_AVAILABLE = False
    print("OR-Tools not available. Install via: pip install ortools")


def generate_synthetic_dataset(num_customers=12, seed=1, area_size=100):
    "\"\"\"Generate a dataset with: ID, x, y, demand, time_window_start, time_window_end, service_time\"\""
    random.seed(seed)
    np.random.seed(seed)
    customers = []
    # Depot at center
    depot = {'id': 0, 'x': area_size/2, 'y': area_size/2, 'demand': 0, 'tw_start': 0, 'tw_end': 1000, 'service': 0}
    customers.append(depot)
    for i in range(1, num_customers+1):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        demand = random.randint(1, 10)
        ready = random.randint(0, 300)
        due = ready + random.randint(60, 300)  # time window width
        service = random.randint(5, 20)
        customers.append({'id': i, 'x': x, 'y': y, 'demand': demand, 'tw_start': ready, 'tw_end': due, 'service': service})
    return customers


def save_dataset_csv(customers, filename='problem2_dataset.csv'):
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','x','y','demand','time_window_start','time_window_end','service_time'])
        for c in customers:
            writer.writerow([c['id'], c['x'], c['y'], c['demand'], c['tw_start'], c['tw_end'], c['service']])


def create_distance_matrix(customers):
    n = len(customers)
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dx = customers[i]['x'] - customers[j]['x']
            dy = customers[i]['y'] - customers[j]['y']
            dist[i,j] = math.hypot(dx, dy)
    return dist


def solve_vrptw_with_ortools(customers, vehicle_count=3, vehicle_capacity=30, time_horizon=2000):
    if not ORTOOLS_AVAILABLE:
        print('OR-Tools not installed. Skipping solver. Dataset saved to results/problem2_dataset.csv')
        return None
    # Prepare data model
    data = {}
    data['locations'] = [(c['x'], c['y']) for c in customers]
    data['num_vehicles'] = vehicle_count
    data['depot'] = 0
    demands = [c['demand'] for c in customers]
    data['demands'] = demands
    data['vehicle_capacities'] = [vehicle_capacity]*vehicle_count
    dist = create_distance_matrix(customers)
    data['distance_matrix'] = (dist).tolist()
    # Time windows
    time_windows = [(int(c['tw_start']), int(c['tw_end'])) for c in customers]
    data['time_windows'] = time_windows
    # Service times
    service_times = [int(c['service']) for c in customers]

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],
        True,  # start cumul to zero
        'Capacity')

    # Add time window constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = int(data['distance_matrix'][from_node][to_node])
        return travel_time + service_times[from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        10000,  # allow waiting time
        time_horizon,
        False,
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    for idx, window in enumerate(data['time_windows']):
        index = manager.NodeToIndex(idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])

    # Setting first solution heuristic and search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(20)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        routes = []
        total_distance = 0
        for v in range(data['num_vehicles']):
            index = routing.Start(v)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, v)
            routes.append(route)
            total_distance += route_distance
        print(f"Found solution. Total distance: {total_distance}")
        # Save route figure
        plot_routes(customers, routes)
        return {'routes': routes, 'total_distance': total_distance}
    else:
        print('No solution found within time limit')
        return None


def plot_routes(customers, routes, filename='results/problem2_routes.png'):
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(6,6))
    colors = ['C'+str(i%10) for i in range(len(routes))]
    for k, route in enumerate(routes):
        xs = [customers[i]['x'] for i in route]
        ys = [customers[i]['y'] for i in route]
        plt.plot(xs, ys, marker='o', label=f'route_{k}', linewidth=1)
        for node in route:
            plt.text(customers[node]['x']+0.5, customers[node]['y']+0.5, str(customers[node]['id']), fontsize=8)
    plt.scatter([customers[0]['x']], [customers[0]['y']], marker='s', color='black', s=80, label='Depot')
    plt.legend()
    plt.title('VRPTW routes')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f'Saved {filename}')


def run_example_and_save():
    customers = generate_synthetic_dataset(num_customers=12, seed=2)
    save_dataset_csv(customers, filename='problem2_dataset.csv')
    result = solve_vrptw_with_ortools(customers, vehicle_count=3, vehicle_capacity=30)
    if result is None:
        print('Solver did not run. Dataset saved to results/problem2_dataset.csv for offline use.')


if __name__ == '__main__':
    run_example_and_save()
