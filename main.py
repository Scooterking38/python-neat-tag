import os
import random
import pickle
import neat
import math
import pygame
import imageio
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor

# -------- SETTINGS --------
CELL_SIZE = 20
GRID_WIDTH = 32
GRID_HEIGHT = 20

MAX_STEPS = 1000
OPPONENTS = 5
GENERATIONS = 30
FPS = 10
PARALLELISM = 4

VIDEO_OUTPUT = "simulation.mp4"
SAVE_DIR = "best_genomes"
os.makedirs(SAVE_DIR, exist_ok=True)

# Headless Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
surface = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))

class SafeStagnation(neat.DefaultStagnation):
    def __init__(self, config, species_set):
        super().__init__(config, species_set)
        self.species_fitness_func = lambda fits: max([f if f is not None else -1e6 for f in fits])
        
# -------- HELPERS --------
def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def move(pos, direction):
    x, y = pos
    old = (x, y)
    if direction == 0: y -= 1
    elif direction == 1: y += 1
    elif direction == 2: x -= 1
    elif direction == 3: x += 1
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return old
    return (x, y)

def draw_frame(chaser, evader):
    surface.fill((0, 0, 0))
    pygame.draw.rect(surface, (255, 0, 0),
                     (chaser[0] * CELL_SIZE, chaser[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(surface, (0, 255, 0),
                     (evader[0] * CELL_SIZE, evader[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return pygame.surfarray.array3d(surface)

def softmax_move(output):
    probs = np.exp(output - np.max(output))
    probs /= np.sum(probs)
    return np.random.choice([0,1,2,3], p=probs)

# -------- CLEAN GENOME --------
def clean_genome(genome):
    """Ensure no None values exist in genome nodes/connections/fintess"""
    if getattr(genome, 'fitness', None) is None:
        genome.fitness = 0.0
    for conn in getattr(genome,'connections',{}).values():
        if getattr(conn,'weight', None) is None: conn.weight = 0.0
        if getattr(conn,'enabled', None) is None: conn.enabled = True
    for node in getattr(genome,'nodes',{}).values():
        if getattr(node,'bias', None) is None: node.bias = 0.0
        if getattr(node,'response', None) is None: node.response = 1.0
    return genome

# -------- SAFE SPECIES FITNESS FUNCTION --------
def safe_species_fitness(fitnesses):
    """NEAT stagnation safe function: replace None with large negative"""
    cleaned = [f if f is not None else -1e6 for f in fitnesses]
    return max(cleaned)

# -------- SIMULATION --------
def simulate_game(net_chaser, net_evader, max_steps=MAX_STEPS):
    pos_chaser = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
    pos_evader = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
    while pos_chaser == pos_evader:
        pos_evader = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

    prev_chaser = deque([0,0], maxlen=2)
    prev_evader = deque([0,0], maxlen=2)
    caught = False
    penalty = 0.0

    for _ in range(max_steps):
        dx = (pos_evader[0] - pos_chaser[0]) / GRID_WIDTH
        dy = (pos_evader[1] - pos_chaser[1]) / GRID_HEIGHT
        dist = math.sqrt(dx*dx + dy*dy)

        # Input: dx, dy, dist, last 2 moves chaser, last 2 moves evader
        input_chaser = [dx, dy, dist] + list(prev_chaser) + list(prev_evader)
        input_evader = [-dx, -dy, dist] + list(prev_evader) + list(prev_chaser)

        move_chaser = softmax_move(net_chaser.activate(input_chaser))
        move_evader = softmax_move(net_evader.activate(input_evader))

        if len(prev_chaser) >= 2 and prev_chaser[-1] == move_chaser: penalty += 0.2
        if len(prev_evader) >= 2 and prev_evader[-1] == move_evader: penalty += 0.2

        prev_chaser.append(move_chaser)
        prev_evader.append(move_evader)

        pos_chaser = move(pos_chaser, move_chaser)
        pos_evader = move(pos_evader, move_evader)

        if pos_chaser == pos_evader:
            caught = True
            break

    final_dist = distance(pos_chaser, pos_evader)

    # Positive rewards: rare, large for catching or escaping
    reward_chaser = 50.0 if caught else 1.0/(final_dist + 1)
    reward_evader = 50.0 if not caught else 1.0/(final_dist + 1)

    reward_chaser -= penalty
    reward_evader -= penalty

    return reward_chaser, reward_evader, caught

# -------- WORKER --------
def simulation_worker(args):
    g1, g2, config1, config2 = args
    try:
        net1 = neat.nn.RecurrentNetwork.create(clean_genome(g1), config1)
        net2 = neat.nn.RecurrentNetwork.create(clean_genome(g2), config2)
        f_ch, f_ev, _ = simulate_game(net1, net2)
        f_ch = f_ch if isinstance(f_ch, (int,float)) else 0.0
        f_ev = f_ev if isinstance(f_ev, (int,float)) else 0.0
        return f_ch, f_ev
    except Exception as e:
        print(f"[WARN] Genome simulation error: {e}")
        return 0.0, 0.0

# -------- EVALUATION --------
def eval_genomes(genomes1, genomes2, config1, config2):
    for _, g in genomes1: g.fitness = 0.0
    for _, g in genomes2: g.fitness = 0.0

    tasks = []
    for _, g1 in genomes1:
        opponents = random.sample(genomes2, min(OPPONENTS, len(genomes2)))
        for _, g2 in opponents:
            tasks.append((g1, g2, config1, config2))

    with ProcessPoolExecutor(max_workers=PARALLELISM) as executor:
        results = list(executor.map(simulation_worker, tasks))

    idx = 0
    for _, g1 in genomes1:
        for _ in range(min(OPPONENTS, len(genomes2))):
            f_ch, f_ev = results[idx]
            g1.fitness += f_ch
            g2_genome = genomes2[idx % len(genomes2)][1]
            if g2_genome.fitness is None: g2_genome.fitness = 0.0
            g2_genome.fitness += f_ev
            idx += 1

    avg_ch = sum(g.fitness for _, g in genomes1) / len(genomes1)
    avg_ev = sum(g.fitness for _, g in genomes2) / len(genomes2)
    print(f"[INFO] Chaser avg fitness: {avg_ch:.2f}, Evader avg fitness: {avg_ev:.2f}")

# -------- SAVE BEST --------
def save_best(pop, name):
    best = max(pop.population.values(), key=lambda g: g.fitness)
    path = os.path.join(SAVE_DIR, f"{name}_best.pkl")
    if os.path.exists(path):
        with open(path,"rb") as f:
            old = pickle.load(f)
        if old.fitness >= best.fitness: return
    with open(path,"wb") as f:
        pickle.dump(best, f)
    print(f"[INFO] Saved best {name} genome")

# -------- VIDEO --------
def record_video(config_path):
    with open(os.path.join(SAVE_DIR,"chaser_best.pkl"),"rb") as f: g1 = pickle.load(f)
    with open(os.path.join(SAVE_DIR,"evader_best.pkl"),"rb") as f: g2 = pickle.load(f)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    net1 = neat.nn.RecurrentNetwork.create(clean_genome(g1), config)
    net2 = neat.nn.RecurrentNetwork.create(clean_genome(g2), config)

    pos1 = (5,5)
    pos2 = (25,15)
    frames = []

    for _ in range(600):
        dx = (pos2[0]-pos1[0])/GRID_WIDTH
        dy = (pos2[1]-pos1[1])/GRID_HEIGHT
        dist = math.sqrt(dx*dx + dy*dy)
        input1 = [dx,dy,dist,0,0,0,0]
        input2 = [-dx,-dy,dist,0,0,0,0]
        move1 = softmax_move(net1.activate(input1))
        move2 = softmax_move(net2.activate(input2))
        pos1 = move(pos1, move1)
        pos2 = move(pos2, move2)
        frames.append(draw_frame(pos1,pos2).swapaxes(0,1))

    imageio.mimsave(VIDEO_OUTPUT, frames, fps=FPS, format="FFMPEG")
    print(f"🎥 Video saved: {VIDEO_OUTPUT}")

# -------- MAIN LOOP --------
def run():
    config_path = "config.txt"
    config1 = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, SafeStagnation, config_path)
    config2 = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, SafeStagnation, config_path)

    pop1 = neat.Population(config1)
    pop2 = neat.Population(config2)

    # --- override stagnation fitness to avoid NoneType errors ---
    pop1.stagnation.species_fitness_func = safe_species_fitness
    pop2.stagnation.species_fitness_func = safe_species_fitness

    # Load best genomes safely
    def load_best(pop, filename):
        path = os.path.join(SAVE_DIR, filename)
        if os.path.exists(path):
            with open(path,"rb") as f:
                best = pickle.load(f)
            best = clean_genome(best)
            keys = list(pop.population.keys())
            pop.population[keys[0]] = best
            for k in keys[1:]:
                clone = pickle.loads(pickle.dumps(best))
                clone.fitness = 0.0
                pop.population[k] = clean_genome(clone)
            print(f"[INFO] Loaded and cleaned {filename}")

    load_best(pop1,"chaser_best.pkl")
    load_best(pop2,"evader_best.pkl")

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen} ---")
        eval_genomes(list(pop1.population.items()), list(pop2.population.items()), config1, config2)
        save_best(pop1,"chaser")
        save_best(pop2,"evader")

        pop1.population = pop1.reproduction.reproduce(config1, pop1.species, config1.pop_size, pop1.generation)
        pop2.population = pop2.reproduction.reproduce(config2, pop2.species, config2.pop_size, pop2.generation)
        pop1.species.speciate(config1, pop1.population, pop1.generation)
        pop2.species.speciate(config2, pop2.population, pop2.generation)
        pop1.generation += 1
        pop2.generation += 1

    record_video(config_path)

if __name__ == "__main__":
    run()
