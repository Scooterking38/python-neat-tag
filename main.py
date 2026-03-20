import neat
import math
import random
import cv2
import numpy as np
from multiprocessing import Pool

# ----------------------------
# Config
# ----------------------------
WIDTH, HEIGHT = 200, 200
MAX_STEPS = 500
SPEED = 5
TAG_REWARD = 100
ROUNDS_PER_MATCH = 500
GENERATIONS = 2000
MIN_MOVE = 1.0          # minimum movement to avoid freeze
INACTIVITY_PENALTY = 50 # heavy punishment for zero movement
PARALLEL = 16
EXPLORATION_BIAS = 0.1  # small bias for early generations
TAGGER_SPEED = 1.0
EVADER_SPEED = 1.5  # evader moves faster than tagger

# ----------------------------
# Safe helpers
# ----------------------------
def safe(x, default=0.0):
    try:
        if x is None or math.isnan(x) or math.isinf(x):
            return default
        return float(x)
    except:
        return default

def safe_output(out):
    if out is None or not hasattr(out, "__len__") or len(out) < 2:
        return 0.5, 0.5
    return max(0, min(1, safe(out[0], 0.5))), max(0, min(1, safe(out[1], 0.5)))

def safe_activate(net, state):
    try:
        if net is None:
            return [0.5, 0.5]
        out = net.activate(state)
        return out if out is not None else [0.5, 0.5]
    except:
        return [0.5, 0.5]

# ----------------------------
# Core classes
# ----------------------------
# Square class with solid walls
class Square:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 10

    def move(self, dx, dy):
        wall_hit = False
        new_x = self.x + dx
        new_y = self.y + dy

        # horizontal walls
        if new_x < 0:
            self.x = 0
            wall_hit = True
        elif new_x + self.size > WIDTH:
            self.x = WIDTH - self.size
            wall_hit = True
        else:
            self.x = new_x

        # vertical walls
        if new_y < 0:
            self.y = 0
            wall_hit = True
        elif new_y + self.size > HEIGHT:
            self.y = HEIGHT - self.size
            wall_hit = True
        else:
            self.y = new_y

        return wall_hit

# Game class
class Game:
    def __init__(self):
        self.tagger = Square(50, 100)
        self.evader = Square(150, 100)

    def get_state(self):
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y
        return [dx / WIDTH, dy / HEIGHT]

    def distance_vector(self):
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y
        return dx, dy

    def step(self, out1, out2, early_gen=True):
        # decode movement
        def decode(o, speed):
            ox, oy = safe_output(o)
            dx = (ox - 0.5) * 2 * speed
            dy = (oy - 0.5) * 2 * speed
            # minimum movement enforcement
            if 0 < abs(dx) < MIN_MOVE:
                dx = MIN_MOVE * (1 if dx > 0 else -1)
            if 0 < abs(dy) < MIN_MOVE:
                dy = MIN_MOVE * (1 if dy > 0 else -1)
            # early generation randomness
            if early_gen:
                dx += random.uniform(-0.1, 0.1)
                dy += random.uniform(-0.1, 0.1)
            return dx, dy

        dx1, dy1 = decode(out1, TAGGER_SPEED)
        dx2, dy2 = decode(out2, EVADER_SPEED)

        hit1 = self.tagger.move(dx1, dy1)
        hit2 = self.evader.move(dx2, dy2)

        # vector from evader → tagger
        vec_x, vec_y = self.tagger.x - self.evader.x, self.tagger.y - self.evader.y
        dist = math.sqrt(vec_x**2 + vec_y**2)
        if dist == 0:
            dist = 1e-5
        ux, uy = vec_x / dist, vec_y / dist

        tagged = dist < (self.tagger.size + self.evader.size)
        return tagged, dx1, dy1, dx2, dy2, hit1, hit2, ux, uy

# Evaluation function
def eval_pair_args(args):
    net_tagger, net_evader, early_gen = args
    score_tagger = 0.0
    score_evader = 0.0

    for _ in range(ROUNDS_PER_MATCH):
        game = Game()
        tagged = False

        for _ in range(MAX_STEPS):
            state = game.get_state()
            out_tagger = safe_output(net_tagger.activate(state) if net_tagger else [0.5,0.5])
            out_evader = safe_output(net_evader.activate(state) if net_evader else [0.5,0.5])

            tagged, dx_t, dy_t, dx_e, dy_e, hit_t, hit_e, ux, uy = game.step(out_tagger, out_evader, early_gen)

            # Project movements along evader→tagger vector
            proj_tagger = dx_t*ux + dy_t*uy   # tagger wants to move closer
            proj_evader  = dx_e*ux + dy_e*uy  # evader movement

            # Rewards
            score_tagger += max(0, proj_tagger) * 0.5
            score_evader += max(0, -proj_evader) * 5  # strong reward for fleeing

            # Punishment for moving toward opponent
            if proj_evader > 0:
                score_evader -= proj_evader * 2

            # Inactivity penalties
            if abs(dx_t)+abs(dy_t) < MIN_MOVE:
                score_tagger -= INACTIVITY_PENALTY
            if abs(dx_e)+abs(dy_e) < MIN_MOVE:
                score_evader -= INACTIVITY_PENALTY

            # Wall penalties
            if hit_t: score_tagger -= 20
            if hit_e: score_evader -= 10

            # Survival bonus
            score_evader += 1.0

            # Tag reward
            if tagged:
                score_tagger += TAG_REWARD
                break

        # bonus for surviving full round
        if not tagged:
            score_evader += MAX_STEPS

    return safe(score_tagger), safe(score_evader)
# ----------------------------
# NEAT training with parallel evaluation
# ----------------------------
def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop_tagger = neat.Population(config)
    pop_evader = neat.Population(config)

    for gen in range(GENERATIONS):
        taggers = list(pop_tagger.population.items())
        evaders = list(pop_evader.population.items())

        for _, g in taggers: g.fitness = 0.0
        for _, g in evaders: g.fitness = 0.0

        genome_pairs = []
        pair_map = []
        early_gen = gen < 50

        for id1, g1 in taggers:
            net1 = neat.nn.FeedForwardNetwork.create(g1, config)
            opponents = random.sample(evaders, min(3, len(evaders)))
            for id2, g2 in opponents:
                net2 = neat.nn.FeedForwardNetwork.create(g2, config)
                genome_pairs.append((net1, net2, early_gen))
                pair_map.append((g1, g2))

        # parallel evaluation
        with Pool(PARALLEL) as pool:
            results = pool.map(eval_pair_args, genome_pairs)

        for (g1, g2), (s1, s2) in zip(pair_map, results):
            g1.fitness += s1
            g2.fitness += s2

        pop_tagger.reporters.post_evaluate(config, pop_tagger.population, pop_tagger.species, None)
        pop_evader.reporters.post_evaluate(config, pop_evader.population, pop_evader.species, None)

        pop_tagger.population = pop_tagger.reproduction.reproduce(config, pop_tagger.species, config.pop_size, gen)
        pop_evader.population = pop_evader.reproduction.reproduce(config, pop_evader.species, config.pop_size, gen)

        pop_tagger.species.speciate(config, pop_tagger.population, gen)
        pop_evader.species.speciate(config, pop_evader.population, gen)

        print(f"Generation {gen} complete")

    best_tagger = max(pop_tagger.population.values(), key=lambda g: safe(g.fitness, -1e9))
    best_evader = max(pop_evader.population.values(), key=lambda g: safe(g.fitness, -1e9))
    return best_tagger, best_evader, config

# ----------------------------
# Rendering
# ----------------------------
def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()
        out1 = net1.activate(state)
        out2 = net2.activate(state)

        # unpack all 7 values; ignore wall hits
        tagged, dx_t, dy_t, dx_e, dy_e, _, _ = game.step(out1, out2, early_gen=False)

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        def draw_square(sq, color):
            x, y, s = int(sq.x), int(sq.y), int(sq.size)
            cv2.rectangle(frame, (x, y), (x + s, y + s), color, -1)

        draw_square(game.tagger, (0, 0, 255))
        draw_square(game.evader, (255, 0, 0))

        frames.append(frame)

        if tagged:
            break

    out = cv2.VideoWriter('best.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (WIDTH, HEIGHT))
    for f in frames:
        out.write(f)
    out.release()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    best_tagger, best_evader, config = run("config.txt")
    net1 = neat.nn.FeedForwardNetwork.create(best_tagger, config)
    net2 = neat.nn.FeedForwardNetwork.create(best_evader, config)
    render_game(net1, net2)
