import numpy as np

np.random.seed(42)
NUM_RESIDENCES = 10
NUM_FACILITIES = 3
GRID_SIZE = 100
residences = np.random.rand(NUM_RESIDENCES, 2) * GRID_SIZE

def objective_function(locs_flat):
    locs = locs_flat.reshape(NUM_FACILITIES, 2)
    distances = np.linalg.norm(residences[:, np.newaxis, :] - locs[np.newaxis, :, :], axis=2)
    return np.mean(np.min(distances, axis=1))

NUM_PARTICLES = 30
MAX_ITER = 80
W, C1, C2 = 0.5, 1.5, 1.5
dim = NUM_FACILITIES * 2

particles_pos = np.random.rand(NUM_PARTICLES, dim) * GRID_SIZE
particles_vel = np.random.uniform(-5, 5, (NUM_PARTICLES, dim))
pbest_pos = np.copy(particles_pos)
pbest_fit = np.array([objective_function(p) for p in particles_pos])

gbest_pos = np.copy(pbest_pos[np.argmin(pbest_fit)])
gbest_fit = np.min(pbest_fit)

print(f"Initial Best Cost (Average Distance): {gbest_fit:.4f}\n")

for iteration in range(MAX_ITER):
    for p in range(NUM_PARTICLES):
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        cognitive = C1 * r1 * (pbest_pos[p] - particles_pos[p])
        social = C2 * r2 * (gbest_pos - particles_pos[p])
        particles_vel[p] = W * particles_vel[p] + cognitive + social
        particles_pos[p] = np.clip(particles_pos[p] + particles_vel[p], 0, GRID_SIZE)

        fit = objective_function(particles_pos[p])
        if fit < pbest_fit[p]:
            pbest_fit[p] = fit
            pbest_pos[p] = np.copy(particles_pos[p])

    best_p = np.argmin(pbest_fit)
    if pbest_fit[best_p] < gbest_fit:
        gbest_fit = pbest_fit[best_p]
        gbest_pos = np.copy(pbest_pos[best_p])

    if (iteration + 1) % 10 == 0 or iteration == 0:
        print(f"Iteration {iteration + 1:02d}/{MAX_ITER}: Best Cost = {gbest_fit:.4f}")


optimal_facilities = gbest_pos.reshape(NUM_FACILITIES, 2)
print("\n--- Optimization Complete ---")
print(f"Final Optimized Average Distance: {gbest_fit:.4f}\n")
print("Optimal Facility Locations:")
for i, loc in enumerate(optimal_facilities, 1):
    print(f"  Facility {i}: X = {loc[0]:.2f}, Y = {loc[1]:.2f}")
