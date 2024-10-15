import csv
import time
from collections import deque
from math import sqrt
import heapq

# --- Functions to read files ---
def read_coordinates(filename):
    coordinates = {}
    # Open the coordinates file and read each line
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Each row contains city name and its coordinates (x, y)
            city, x, y = row[0], float(row[1]), float(row[2])
            coordinates[city] = (x, y)  # Store coordinates in a dictionary
    return coordinates

def read_adjacencies(filename):
    adjacencies = {}
    # Open the adjacencies file and read each line
    with open(filename, 'r') as file:
        for line in file:
            # Each line contains two cities connected by an edge
            city1, city2 = line.strip().split()
            if city1 not in adjacencies:
                adjacencies[city1] = []  # Initialize list if city1 isn't in the dictionary
            if city2 not in adjacencies:
                adjacencies[city2] = []  # Initialize list if city2 isn't in the dictionary
            adjacencies[city1].append(city2)  # Add city2 as a neighbor of city1
            adjacencies[city2].append(city1)  # Add city1 as a neighbor of city2
    return adjacencies

# --- Graph class ---
class Graph:
    def __init__(self, adjacencies):
        self.graph = adjacencies  # The graph is represented as an adjacency list

    # Get all neighbors (adjacent cities) of a city (node)
    def get_neighbors(self, node):
        return self.graph.get(node, [])

# --- Search algorithms with path tracking ---
# Reconstruct the path from start to goal using the 'came_from' dictionary
def reconstruct_path(came_from, start, goal):
    path = [goal]  # Start from the goal and work backwards
    while path[-1] != start:  # Stop when we reach the start
        path.append(came_from[path[-1]])  # Add the previous city in the path
    path.reverse()  # Reverse the path to get the correct order
    return path

# --- Breadth-First Search (BFS) ---
def bfs(graph, start, goal):
    visited = set()  # Set to track visited cities
    queue = deque([start])  # Queue for BFS
    came_from = {start: None}  # Track the path

    while queue:
        city = queue.popleft()  # Pop from the front of the queue (FIFO)
        if city == goal:  # If we reached the goal, reconstruct the path
            return reconstruct_path(came_from, start, goal)

        if city not in visited:
            visited.add(city)  # Mark the city as visited
            for neighbor in graph.get_neighbors(city):
                if neighbor not in visited and neighbor not in came_from:
                    came_from[neighbor] = city  # Record the path
                    queue.append(neighbor)  # Add neighbors to the queue

    return None  # Return None if no path is found

# --- Depth-First Search (DFS) ---
def dfs(graph, start, goal):
    visited = set()  # Set to track visited cities
    stack = [start]  # Stack for DFS
    came_from = {start: None}  # Track the path

    while stack:
        city = stack.pop()  # Pop from the top of the stack (LIFO)
        if city == goal:  # If we reached the goal, reconstruct the path
            return reconstruct_path(came_from, start, goal)

        if city not in visited:
            visited.add(city)  # Mark the city as visited
            for neighbor in graph.get_neighbors(city):
                if neighbor not in visited and neighbor not in came_from:
                    came_from[neighbor] = city  # Record the path
                    stack.append(neighbor)  # Add neighbors to the stack

    return None  # Return None if no path is found

# --- Iterative Deepening Depth-First Search (IDDFS) ---
def dls(graph, node, goal, depth, visited, came_from):
    if depth == 0 and node == goal:  # If depth is 0 and we are at the goal, we found the path
        return True
    if depth > 0:
        visited.add(node)  # Mark the city as visited
        for neighbor in graph.get_neighbors(node):
            if neighbor not in visited:
                came_from[neighbor] = node  # Record the path
                if dls(graph, neighbor, goal, depth - 1, visited, came_from):
                    return True  # Recursively call DLS with reduced depth
    return False

# Perform IDDFS with increasing depth limits
def iddfs(graph, start, goal, max_depth):
    for depth in range(max_depth):
        visited = set()  # Set to track visited cities
        came_from = {start: None}  # Track the path
        if dls(graph, start, goal, depth, visited, came_from):
            return reconstruct_path(came_from, start, goal)
    return None  # Return None if no path is found

# --- Euclidean distance for heuristic (used in A* and Best-First) ---
def euclidean_distance(c1, c2):
    return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)  # Calculate straight-line distance

# --- Best-First Search ---
def best_first_search(graph, coordinates, start, goal):
    open_list = []
    heapq.heappush(open_list, (euclidean_distance(coordinates[start], coordinates[goal]), start))  # Priority queue
    came_from = {start: None}  # Track the path
    visited = set()  # Set to track visited cities

    while open_list:
        _, current = heapq.heappop(open_list)  # Get the city with the lowest heuristic estimate

        if current == goal:  # If we reached the goal, reconstruct the path
            return reconstruct_path(came_from, start, goal)

        visited.add(current)  # Mark the city as visited

        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                came_from[neighbor] = current  # Record the path
                heapq.heappush(open_list, (euclidean_distance(coordinates[neighbor], coordinates[goal]), neighbor))  # Add neighbor to the priority queue

    return None  # Return None if no path is found

# --- A* Search ---
def a_star(graph, coordinates, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue for A*
    came_from = {start: None}  # Track the path
    g_score = {start: 0}  # Cost from start to a given city
    f_score = {start: euclidean_distance(coordinates[start], coordinates[goal])}  # Estimate total cost

    while open_list:
        _, current = heapq.heappop(open_list)  # Get the city with the lowest f_score

        if current == goal:  # If we reached the goal, reconstruct the path
            return reconstruct_path(came_from, start, goal)

        for neighbor in graph.get_neighbors(current):
            tentative_g_score = g_score[current] + euclidean_distance(coordinates[current], coordinates[neighbor])
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current  # Record the path
                g_score[neighbor] = tentative_g_score  # Update g_score
                f_score[neighbor] = g_score[neighbor] + euclidean_distance(coordinates[neighbor], coordinates[goal])
                heapq.heappush(open_list, (f_score[neighbor], neighbor))  # Add neighbor to the priority queue

    return None  # Return None if no path is found

# --- Utility function to measure precise time ---
def measure_time(func, *args):
    start_time = time.perf_counter()  # Higher-resolution timer
    result = func(*args)  # Run the search algorithm
    end_time = time.perf_counter()  # End the timer
    return result, end_time - start_time  # Return result and time taken

# --- Calculate total distance of the route ---
def calculate_total_distance(route, coordinates):
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = coordinates[route[i]]
        city2 = coordinates[route[i + 1]]
        total_distance += euclidean_distance(city1, city2)  # Calculate and accumulate the distance
    return total_distance

# --- Menu and user interaction ---
def menu():
    print("\nSelect a search algorithm:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Iterative Deepening Depth-First Search (IDDFS)")
    print("4. Best-First Search")
    print("5. A* Search")
    print("6. Quit")
    choice = input("Enter the number of your choice: ")
    return choice

# Prompt user for city input, ensuring valid city
def get_city(prompt, coordinates):
    while True:
        city = input(prompt)
        if city in coordinates:
            return city  # Return the city if it's in the dataset
        print("City not found. Please try again.")

# --- Main program ---
def main():
    coordinates = read_coordinates("coordinates.csv")  # Load city coordinates
    adjacencies = read_adjacencies("Adjacencies.txt")  # Load adjacency list
    graph = Graph(adjacencies)  # Initialize the graph

    start_city = get_city("Enter the start city: ", coordinates)  # Get start city from user
    end_city = get_city("Enter the destination city: ", coordinates)  # Get destination city from user

    while True:
        choice = menu()  # Display menu and get algorithm choice

        # Select the search algorithm based on user's choice
        if choice == '1':
            search_func = bfs
        elif choice == '2':
            search_func = dfs
        elif choice == '3':
            # Ask for the max depth when using IDDFS
            max_depth = int(input("Enter the maximum depth for IDDFS: "))  # Get max depth from user
            search_func = lambda g, s, e: iddfs(g, s, e, max_depth)  # Use lambda to pass max_depth
        elif choice == '4':
            search_func = a_star
        elif choice == '5':
            search_func = best_first_search
        elif choice == '6':  # Quit the program
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        # Handle different argument requirements for A* and Best-First
        if choice in ['4', '5']:
            result, time_taken = measure_time(search_func, graph, coordinates, start_city, end_city)
        else:
            result, time_taken = measure_time(search_func, graph, start_city, end_city)

        # Display the result
        if result:
            total_distance = calculate_total_distance(result, coordinates)  # Calculate total distance
            print(f"\nRoute found: {' -> '.join(result)}")
            print(f"Total Distance: {total_distance:.4f} units")
            print(f"Time taken: {time_taken:.8f} seconds\n")
        else:
            print("\nNo route found.\n")


if __name__ == "__main__":
    main()  # Run the main program
