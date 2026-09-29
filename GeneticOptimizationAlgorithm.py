import random
import math

# -----------------------------
# Problem Setup
# -----------------------------

GRID_SIZE = 10

START = (0, 0)
GOAL = (9, 9)

# Obstacles
OBSTACLES = {
    (2, 2), (2, 3), (2, 4),
    (4, 5), (5, 5), (6, 5),
    (7, 2), (8, 2)
}

# Possible robot movements
MOVES = [
    (0, 1),    # Up
    (0, -1),   # Down
    (1, 0),    # Right
    (-1, 0)    # Left
]


# -----------------------------
# Generate a Random Route
# -----------------------------

def generate_route(length):
    return [random.choice(MOVES) for _ in range(length)]


# -----------------------------
# Simulate Route
# -----------------------------

def simulate_route(route):
    position = START
    path = [position]

    for move in route:
        new_position = (
            position[0] + move[0],
            position[1] + move[1]
        )

        # Check boundaries
        if not (
            0 <= new_position[0] < GRID_SIZE and
            0 <= new_position[1] < GRID_SIZE
        ):
            return path, False

        # Check obstacles
        if new_position in OBSTACLES:
            return path, False

        position = new_position
        path.append(position)

        # Stop if destination reached
        if position == GOAL:
            return path, True

    return path, position == GOAL


# -----------------------------
# Fitness Function
# -----------------------------

def fitness(route):
    path, reached_goal = simulate_route(route)

    current = path[-1]

    # Manhattan distance to goal
    distance = abs(GOAL[0] - current[0]) + abs(GOAL[1] - current[1])

    # Route length
    length = len(path)

    # Large penalty if goal isn't reached
    if not reached_goal:
        return 1000 + distance * 10 + length

    # Shorter routes are better
    return length


# -----------------------------
# Selection
# -----------------------------

def selection(population):
    population.sort(key=fitness)

    # Keep the best 20%
    elite_count = max(2, len(population) // 5)

    return population[:elite_count]


# -----------------------------
# Crossover
# -----------------------------

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)

    child = parent1[:point] + parent2[point:]

    return child


# -----------------------------
# Mutation
# -----------------------------

def mutate(route, mutation_rate=0.05):

    for i in range(len(route)):

        if random.random() < mutation_rate:
            route[i] = random.choice(MOVES)

    return route


# -----------------------------
# Genetic Algorithm
# -----------------------------

def genetic_algorithm(
    population_size=200,
    route_length=40,
    generations=500,
    mutation_rate=0.05
):

    # Initial population
    population = [
        generate_route(route_length)
        for _ in range(population_size)
    ]

    for generation in range(generations):

        # Select best individuals
        parents = selection(population)

        new_population = parents.copy()

        # Generate new population
        while len(new_population) < population_size:

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = crossover(parent1, parent2)

            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population

        # Best route
        best_route = min(population, key=fitness)

        path, reached = simulate_route(best_route)

        if reached:
            print(
                f"Generation {generation}: "
                f"Route found with length {len(path)}"
            )

            # Return immediately when a valid route is found
            return best_route, path

    # Return best route after all generations
    best_route = min(population, key=fitness)

    return best_route, simulate_route(best_route)[0]


# -----------------------------
# Run Algorithm
# -----------------------------

best_route, best_path = genetic_algorithm()

print("\nBest Route:")
print(best_path)

print("\nRoute Length:")
print(len(best_path))

print("\nGoal Reached:")
print(best_path[-1] == GOAL)
