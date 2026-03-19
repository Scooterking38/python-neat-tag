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
MAX_STEPS = 20
OPPONENTS = 5  # number of opponents sampled per genome

SAVE_DIR = "best_genomes"
os.makedirs(SAVE_DIR, exist_ok=True)

pygame.init()
win = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))


# -------- HELPER --------
def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move(pos, direction):
    x, y = pos
    old = (x, y)

    if direction == 0:
        y -= 1
    elif direction == 1:
        y += 1
    elif direction == 2:
        x -= 1
    elif direction == 3:
        x += 1

    # detect invalid move
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return old, True

    return (x, y), False


# -------- EVALUATION --------
def eval_genomes(genomes1, genomes2, config1, config2):
    # reset fitness
    for _, g in genomes1:
        g.fitness = 0.0
    for _, g in genomes2:
        g.fitness = 0.0

    for _, g1 in genomes1:
        net1 = neat.nn.FeedForwardNetwork.create(g1, config1)

        opponents = random.sample(genomes2, min(OPPONENTS, len(genomes2)))

        for _, g2 in opponents:
            net2 = neat.nn.FeedForwardNetwork.create(g2, config2)

            # random start positions
            pos1 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

            for step in range(MAX_STEPS):
                dx = (pos2[0] - pos1[0]) / GRID_WIDTH
                dy = (pos2[1] - pos1[1]) / GRID_HEIGHT
                dist = math.sqrt(dx * dx + dy * dy)

                # Chaser
                out1 = net1.activate((dx, dy, dist))
                move1 = out1.index(max(out1))

                # Evader
                out2 = net2.activate((-dx, -dy, dist))
                move2 = out2.index(max(out2))

                pos1, bad1 = move(pos1, move1)
                pos2, bad2 = move(pos2, move2)

                d = distance(pos1, pos2)

                # normalized fitness
                g1.fitness += ((1 / (d + 1)) * 2) / (MAX_STEPS * OPPONENTS)
                g2.fitness += (d / (GRID_WIDTH + GRID_HEIGHT)) / (MAX_STEPS * OPPONENTS)

                # penalties
                if bad1:
                    g1.fitness -= 0.01
                if bad2:
                    g2.fitness -= 0.01

                # survival bonus for evader
                g2.fitness += 0.001

                # catch condition
                if d == 0:
                    g1.fitness += 1
                    g2.fitness -= 1
                    break


# -------- SAVE BEST --------
def save_best(pop, name):
    valid = [g for g in pop.population.values() if g.fitness is not None]

    if not valid:
        print(f"No valid genomes to save for {name}")
        return

    best = max(valid, key=lambda g: g.fitness)

    path = os.path.join(SAVE_DIR, f"{name}_best.pkl")

    # keep global best
    if os.path.exists(path):
        with open(path, "rb") as f:
            old = pickle.load(f)
        if old.fitness >= best.fitness:
            return

    with open(path, "wb") as f:
        pickle.dump(best, f)

    print(f"Saved NEW best {name} genome → {path}")


# -------- RUN --------
def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config1 = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    config2 = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop1 = neat.Population(config1)  # chaser
    pop2 = neat.Population(config2)  # evader

    GENERATIONS = 10

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen} ---")

        genomes1 = list(pop1.population.items())
        genomes2 = list(pop2.population.items())

        # evaluate
        eval_genomes(genomes1, genomes2, config1, config2)

        # ✅ SAVE BEFORE reproduction
        save_best(pop1, "chaser")
        save_best(pop2, "evader")

        # reporting
        pop1.reporters.post_evaluate(config1, pop1.population, pop1.species, None)
        pop2.reporters.post_evaluate(config2, pop2.population, pop2.species, None)

        # reproduce
        pop1.population = pop1.reproduction.reproduce(
            config1, pop1.species, config1.pop_size, pop1.generation
        )
        pop2.population = pop2.reproduction.reproduce(
            config2, pop2.species, config2.pop_size, pop2.generation
        )

        # speciate
        pop1.species.speciate(config1, pop1.population, pop1.generation)
        pop2.species.speciate(config2, pop2.population, pop2.generation)

        pop1.generation += 1
        pop2.generation += 1


if __name__ == "__main__":
    run()
