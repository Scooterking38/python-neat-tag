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
GENERATIONS = 200
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
class Square:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 10

    def move(self, dx, dy):
        """
        Moves the square, enforcing solid walls.
        Returns True if a wall was hit.
        """
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


class Game:
    def __init__(self):
        self.tagger = Square(50, 100)
        self.evader = Square(150, 100)

    def get_state(self):
        # only dx, dy
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y
        return [dx / WIDTH, dy / HEIGHT]

    def distance_vector(self):
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y
        return dx, dy

    def step(self, out1, out2, early_gen=True):
        """
        Moves both agents.
        Returns:
        tagged, dx1, dy1, dx2, dy2, tagger_hit_wall, evader_hit_wall
        """
        def decode(o, speed):
            ox, oy = safe_output(o)
            dx = (ox - 0.5) * 2 * speed
            dy = (oy - 0.5) * 2 * speed

            # minimum movement enforcement
            if 0 < abs(dx) < MIN_MOVE:
                dx = MIN_MOVE * (1 if dx > 0 else -1)
            if 0 < abs(dy) < MIN_MOVE:
                dy = MIN_MOVE * (1 if dy > 0 else -1)

            # small exploration noise
            dx += random.uniform(-0.1, 0.1)
            dy += random.uniform(-0.1, 0.1)

            return dx, dy

        dx1, dy1 = decode(out1, TAGGER_SPEED)
        dx2, dy2 = decode(out2, EVADER_SPEED)

        hit1 = self.tagger.move(dx1, dy1)
        hit2 = self.evader.move(dx2, dy2)

        # compute distance to see if tagged
        dx, dy = self.distance_vector()
        dist = math.sqrt(dx**2 + dy**2)
        tagged = dist < (self.tagger.size + self.evader.size)

        return tagged, dx1, dy1, dx2, dy2, hit1, hit2
# ----------------------------
# Evaluate a single genome pair
# ----------------------------
def eval_pair_args(args):
    """
    Evaluate a single tagger/evader neural network pair.
    Returns: (tagger_score, evader_score)
    Features:
    - Solid walls
    - Minimum movement enforced
    - Tagger rewarded for moving closer
    - Evader rewarded for moving away
    - Evader heavily but moderately punished for moving toward tagger
    - Wall-hit penalties
    - Continuous survival reward for evader
    """

    net_tagger, net_evader, early_gen = args
    score_tagger = 0.0
    score_evader = 0.0

    for _ in range(ROUNDS_PER_MATCH):
        game = Game()
        tagged = False

        for _ in range(MAX_STEPS):
            # Get dx, dy input state
            state = game.get_state()

            # Network outputs
            out_tagger = safe_activate(net_tagger, state)
            out_evader = safe_activate(net_evader, state)

            # Step game
            tagged, dx_t, dy_t, dx_e, dy_e, hit_t, hit_e = game.step(out_tagger, out_evader, early_gen)

            # Vector from tagger to evader
            vec_x, vec_y = game.distance_vector()
            dist = math.sqrt(vec_x**2 + vec_y**2)
            if dist == 0:
                dist = 1e-5  # avoid divide by zero
            ux, uy = vec_x / dist, vec_y / dist

            # Project movements onto the tagger→evader vector
            proj_tagger = dx_t * ux + dy_t * uy
            proj_evader  = dx_e * ux + dy_e * uy

            # --- directional rewards ---
            score_tagger += max(0, proj_tagger) * 0.5   # slower, incremental reward for closing
            score_evader  += max(0, -proj_evader) * 5   # strong reward for fleeing

            # --- moderate punishment for evader moving toward tagger ---
            if proj_evader > 0:
                score_evader -= proj_evader * 2

            # --- punish inactivity ---
            if abs(dx_t) + abs(dy_t) < MIN_MOVE:
                score_tagger -= INACTIVITY_PENALTY
            if abs(dx_e) + abs(dy_e) < MIN_MOVE:
                score_evader -= INACTIVITY_PENALTY

            # --- punish wall hits ---
            if hit_t:
                score_tagger -= 20
            if hit_e:
                score_evader -= 10   # slightly less harsh to let evader explore edges

            # --- continuous survival reward for evader ---
            score_evader += 1.0

            # --- tag reward ---
            if tagged:
                score_tagger += TAG_REWARD
                break

        # Bonus for surviving full round
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
