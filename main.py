import neat
import random
import math
import cv2
import numpy as np

WIDTH, HEIGHT = 200, 200
MAX_STEPS = 500
SPEED = 3

TAG_REWARD = 100
ROUNDS_PER_MATCH = 5
GENERATIONS = 200


# ----------------------------
# Safety helpers
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

    x = safe(out[0], 0.5)
    y = safe(out[1], 0.5)

    return max(0, min(1, x)), max(0, min(1, y))


def safe_activate(net, state):
    try:
        if net is None:
            return [0.5, 0.5]
        out = net.activate(state)
        return out if out is not None else [0.5, 0.5]
    except:
        return [0.5, 0.5]


# ----------------------------
# Core
# ----------------------------

class Square:
    def __init__(self, x, y):
        self.x = safe(x)
        self.y = safe(y)
        self.size = 10

    def move(self, dx, dy):
        self.x = (self.x + safe(dx)) % WIDTH
        self.y = (self.y + safe(dy)) % HEIGHT


class Game:
    def __init__(self):
        self.tagger = Square(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        self.evader = Square(random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def get_state(self):
        dx = self.evader.x - self.tagger.x
        dy = self.evader.y - self.tagger.y

        if dx > WIDTH/2: dx -= WIDTH
        if dx < -WIDTH/2: dx += WIDTH
        if dy > HEIGHT/2: dy -= HEIGHT
        if dy < -HEIGHT/2: dy += HEIGHT

        return [dx / WIDTH, dy / HEIGHT]

    def distance(self):
        dx = min(abs(self.tagger.x - self.evader.x), WIDTH - abs(self.tagger.x - self.evader.x))
        dy = min(abs(self.tagger.y - self.evader.y), HEIGHT - abs(self.tagger.y - self.evader.y))
        return math.sqrt(dx*dx + dy*dy)

    def step(self, out1, out2):
        def decode(o):
            ox, oy = safe_output(o)
            return (ox - 0.5) * 2 * SPEED, (oy - 0.5) * 2 * SPEED

        dx1, dy1 = decode(out1)
        dx2, dy2 = decode(out2)

        self.tagger.move(dx1, dy1)
        self.evader.move(dx2, dy2)

        dist = self.distance()
        tagged = dist < (self.tagger.size + self.evader.size)

        return tagged, dist, dx1, dy1, dx2, dy2


# ----------------------------
# Evaluation
# ----------------------------

def eval_pair(net_tagger, net_evader):
    score_tagger = 0.0
    score_evader = 0.0

    for _ in range(ROUNDS_PER_MATCH):
        game = Game()
        prev_dist = game.distance()

        tagged = False

        for step in range(MAX_STEPS):
            state = game.get_state()

            out1 = safe_activate(net_tagger, state)
            out2 = safe_activate(net_evader, state)

            tagged, dist, dx1, dy1, dx2, dy2 = game.step(out1, out2)

            dist = safe(dist)
            prev_dist = safe(prev_dist)

            # --- TAGGER: get closer ---
            score_tagger += 2 * max(0, prev_dist - dist)

            # --- EVADER: move away ---
            score_evader += 2 * max(0, dist - prev_dist)

            # --- inactivity penalty (CRITICAL) ---
            if abs(dx1) + abs(dy1) < 0.01:
                score_tagger -= 0.1
            if abs(dx2) + abs(dy2) < 0.01:
                score_evader -= 0.1

            # --- evader survival reward ---
            score_evader += 0.1

            # --- tag ---
            if tagged:
                score_tagger += TAG_REWARD
                break

            prev_dist = dist

        if not tagged:
            score_evader += MAX_STEPS

    return safe(score_tagger), safe(score_evader)


# ----------------------------
# Training
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
        taggers = list(pop_tagger.population.items())
        evaders = list(pop_evader.population.items())

        for _, g in taggers:
            g.fitness = 0.0
        for _, g in evaders:
            g.fitness = 0.0

        for id1, g1 in taggers:
            net1 = neat.nn.FeedForwardNetwork.create(g1, config)

            opponents = random.sample(evaders, min(3, len(evaders)))

            for id2, g2 in opponents:
                net2 = neat.nn.FeedForwardNetwork.create(g2, config)

                s1, s2 = eval_pair(net1, net2)

                g1.fitness += safe(s1)
                g2.fitness += safe(s2)

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

    best_tagger = max(pop_tagger.population.values(), key=lambda g: safe(g.fitness, -1e9))
    best_evader = max(pop_evader.population.values(), key=lambda g: safe(g.fitness, -1e9))

    return best_tagger, best_evader, config


# ----------------------------
# Render
# ----------------------------

def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()

        out1 = safe_activate(net1, state)
        out2 = safe_activate(net2, state)

        tagged, _, _, _, _, _ = game.step(out1, out2)

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        def draw(sq, color):
            x, y = int(sq.x), int(sq.y)
            s = int(sq.size)
            cv2.rectangle(frame, (x, y), (x + s, y + s), color, -1)

        draw(game.tagger, (0, 0, 255))
        draw(game.evader, (255, 0, 0))

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
