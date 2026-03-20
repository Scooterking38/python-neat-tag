import neat
import random
import math
import cv2
import numpy as np

WIDTH, HEIGHT = 200, 200
MAX_STEPS = 500
SPEED = 1

TAG_REWARD = 100
ROUNDS_PER_MATCH = 5
GENERATIONS = 200


# ----------------------------
# Utility (SAFE)
# ----------------------------

def safe_number(x, default=0.0):
    if x is None:
        return default
    try:
        if math.isnan(x) or math.isinf(x):
            return default
        return float(x)
    except:
        return default


def safe_output(out):
    if out is None or not hasattr(out, "__len__") or len(out) < 2:
        return 0.0, 0.0

    x = safe_number(out[0], 0.5)
    y = safe_number(out[1], 0.5)

    # clamp
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    return x, y


# ----------------------------
# Core Classes
# ----------------------------

class Square:
    def __init__(self, x, y):
        self.x = safe_number(x)
        self.y = safe_number(y)
        self.size = 10

    def move(self, dx, dy):
        dx = safe_number(dx)
        dy = safe_number(dy)

        self.x = (self.x + dx) % WIDTH
        self.y = (self.y + dy) % HEIGHT


class Game:
    def __init__(self):
        self.tagger = Square(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        self.evader = Square(random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def get_state(self):
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y

        # wrap-aware shortest direction
        if dx > WIDTH / 2:
            dx -= WIDTH
        elif dx < -WIDTH / 2:
            dx += WIDTH

        if dy > HEIGHT / 2:
            dy -= HEIGHT
        elif dy < -HEIGHT / 2:
            dy += HEIGHT

        return [
            dx / WIDTH,
            dy / HEIGHT
        ]

    def step(self, out1, out2):
        def decode(o):
            ox, oy = safe_output(o)
            return (ox - 0.5) * 2 * SPEED, (oy - 0.5) * 2 * SPEED

        dx1, dy1 = decode(out1)
        dx2, dy2 = decode(out2)

        self.tagger.move(dx1, dy1)
        self.evader.move(dx2, dy2)

        dx = min(abs(self.tagger.x - self.evader.x), WIDTH - abs(self.tagger.x - self.evader.x))
        dy = min(abs(self.tagger.y - self.evader.y), HEIGHT - abs(self.tagger.y - self.evader.y))

        dist = math.sqrt(dx * dx + dy * dy)

        tagged = dist < (self.tagger.size + self.evader.size)

        return tagged, dist


# ----------------------------
# Evaluation
# ----------------------------

def safe_activate(net, state):
    try:
        if net is None:
            return [0.5, 0.5]
        out = net.activate(state)
        return out if out is not None else [0.5, 0.5]
    except:
        return [0.5, 0.5]


def eval_pair(net_tagger, net_evader):
    score_tagger = 0.0
    score_evader = 0.0

    for _ in range(ROUNDS_PER_MATCH):
        game = Game()

        # initial distance
        _, prev_dist = game.step([0.5, 0.5], [0.5, 0.5])

        tagged = False

        for step in range(MAX_STEPS):
            state = game.get_state()

            out1 = safe_activate(net_tagger, state)
            out2 = safe_activate(net_evader, state)

            tagged, dist = game.step(out1, out2)

            prev_dist = safe_number(prev_dist)
            dist = safe_number(dist)

            # --- TAGGER ---
            score_tagger += max(0.0, prev_dist - dist)

            # --- EVADER ---
            score_evader += max(0.0, dist - prev_dist)

            prev_dist = dist

            if tagged:
                score_tagger += TAG_REWARD
                break

        if not tagged:
            score_evader += MAX_STEPS

    return safe_number(score_tagger), safe_number(score_evader)


# ----------------------------
# Training Loop
# ----------------------------

def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop_tagger = neat.Population(config)
    pop_evader = neat.Population(config)

    for gen in range(GENERATIONS):
        tagger_genomes = list(pop_tagger.population.items())
        evader_genomes = list(pop_evader.population.items())

        for _, g in tagger_genomes:
            g.fitness = 0.0

        for _, g in evader_genomes:
            g.fitness = 0.0

        for id1, g1 in tagger_genomes:
            net1 = neat.nn.FeedForwardNetwork.create(g1, config)

            opponents = random.sample(
                evader_genomes,
                min(3, len(evader_genomes))
            )

            for id2, g2 in opponents:
                net2 = neat.nn.FeedForwardNetwork.create(g2, config)

                s1, s2 = eval_pair(net1, net2)

                g1.fitness += safe_number(s1)
                g2.fitness += safe_number(s2)

        pop_tagger.reporters.post_evaluate(config, pop_tagger.population, pop_tagger.species, None)
        pop_evader.reporters.post_evaluate(config, pop_evader.population, pop_evader.species, None)

        pop_tagger.population = pop_tagger.reproduction.reproduce(
            config, pop_tagger.species, config.pop_size, gen
        )
        pop_evader.population = pop_evader.reproduction.reproduce(
            config, pop_evader.species, config.pop_size, gen
        )

        pop_tagger.species.speciate(config, pop_tagger.population, gen)
        pop_evader.species.speciate(config, pop_evader.population, gen)

        print(f"Generation {gen} complete")

    best_tagger = max(
        pop_tagger.population.values(),
        key=lambda g: safe_number(g.fitness, -1e9)
    )

    best_evader = max(
        pop_evader.population.values(),
        key=lambda g: safe_number(g.fitness, -1e9)
    )

    return best_tagger, best_evader, config


# ----------------------------
# Rendering
# ----------------------------

def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()

        out1 = safe_activate(net1, state)
        out2 = safe_activate(net2, state)

        tagged, _ = game.step(out1, out2)

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        def draw_square(sq, color):
            x, y = int(sq.x), int(sq.y)
            s = int(sq.size)
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

if __name__ == '__main__':
    best_tagger, best_evader, config = run('config.txt')

    net1 = neat.nn.FeedForwardNetwork.create(best_tagger, config)
    net2 = neat.nn.FeedForwardNetwork.create(best_evader, config)

    render_game(net1, net2)
