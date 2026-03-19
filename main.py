import os
import random
import pickle
import pygame
import neat
import math

# -------- SETTINGS --------
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
MAX_STEPS = 200

SAVE_DIR = "best_genomes"
os.makedirs(SAVE_DIR, exist_ok=True)

pygame.init()
win = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))

# -------- HELPER --------
def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move(pos, direction):
    x, y = pos
    if direction == 0: y -= 1
    elif direction == 1: y += 1
    elif direction == 2: x -= 1
    elif direction == 3: x += 1

    x = max(0, min(GRID_WIDTH - 1, x))
    y = max(0, min(GRID_HEIGHT - 1, y))
    return x, y


# -------- EVALUATION --------
def eval_genomes(genomes1, genomes2, config1, config2):
    for _, g in genomes1:
        g.fitness = 0
    for _, g in genomes2:
        g.fitness = 0

    for _, g1 in genomes1:
        net1 = neat.nn.FeedForwardNetwork.create(g1, config1)

        for _, g2 in genomes2:
            net2 = neat.nn.FeedForwardNetwork.create(g2, config2)

            # random start positions
            pos1 = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            pos2 = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

            for step in range(MAX_STEPS):
                dx = (pos2[0] - pos1[0]) / GRID_WIDTH
                dy = (pos2[1] - pos1[1]) / GRID_HEIGHT

                # Chaser (green)
                out1 = net1.activate((dx, dy))
                move1 = out1.index(max(out1))

                # Evader (red)
                out2 = net2.activate((-dx, -dy))
                move2 = out2.index(max(out2))

                pos1 = move(pos1, move1)
                pos2 = move(pos2, move2)

                d = distance(pos1, pos2)

                # Fitness shaping
                g1.fitness += (1 / (d + 1)) * 2  # closer = better
                g2.fitness += (d / (GRID_WIDTH + GRID_HEIGHT))  # farther = better

                # Catch condition
                if d == 0:
                    g1.fitness += 50
                    g2.fitness -= 50
                    break


# -------- SAVE BEST --------
def save_best(pop, name):
    best = max(pop.population.values(), key=lambda g: g.fitness)
    path = os.path.join(SAVE_DIR, f"{name}_best.pkl")
    with open(path, "wb") as f:
        pickle.dump(best, f)
    print(f"Saved best {name} genome → {path}")


# -------- RUN --------
def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config1 = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    config2 = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop1 = neat.Population(config1)  # chaser
    pop2 = neat.Population(config2)  # evader

    GENERATIONS = 10

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen} ---")

        genomes1 = list(pop1.population.items())
        genomes2 = list(pop2.population.items())

        eval_genomes(genomes1, genomes2, config1, config2)

        pop1.reporters.post_evaluate(config1, pop1.population, pop1.species, None)
        pop2.reporters.post_evaluate(config2, pop2.population, pop2.species, None)

        pop1.population = pop1.reproduction.reproduce(
            config1, pop1.species, config1.pop_size, pop1.generation
        )
        pop2.population = pop2.reproduction.reproduce(
            config2, pop2.species, config2.pop_size, pop2.generation
        )

        pop1.species.speciate(config1, pop1.population, pop1.generation)
        pop2.species.speciate(config2, pop2.population, pop2.generation)

        pop1.generation += 1
        pop2.generation += 1

        # Save best each generation
        save_best(pop1, "chaser")
        save_best(pop2, "evader")


if __name__ == "__main__":
    run()
