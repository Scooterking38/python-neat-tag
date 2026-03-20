import neat
import random
import cv2
import numpy as np

WIDTH, HEIGHT = 200, 200
MAX_STEPS = 300
SPEED = 3

WRAP_PENALTY = 0.2
TAG_REWARD = 5
EVADE_REWARD = 3
EVADE_PUNISH = 3

ROUNDS_PER_MATCH = 5
GENERATIONS = 30


class Square:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.size = 10

    def move(self, dx, dy):
        wall_hit = False

        new_x = self.x + dx
        new_y = self.y + dy

        if new_x < 0 or new_x >= WIDTH:
            wall_hit = True
        if new_y < 0 or new_y >= HEIGHT:
            wall_hit = True

        self.x = new_x % WIDTH
        self.y = new_y % HEIGHT

        return wall_hit


class Game:
    def __init__(self):
        self.tagger = Square()
        self.evader = Square()

    def step(self, out1, out2):
        def decode(o):
            return (o[0] - 0.5) * 2 * SPEED, (o[1] - 0.5) * 2 * SPEED

        dx1, dy1 = decode(out1)
        dx2, dy2 = decode(out2)

        hit1 = self.tagger.move(dx1, dy1)
        hit2 = self.evader.move(dx2, dy2)

        dx = min(abs(self.tagger.x - self.evader.x), WIDTH - abs(self.tagger.x - self.evader.x))
        dy = min(abs(self.tagger.y - self.evader.y), HEIGHT - abs(self.tagger.y - self.evader.y))
        dist = (dx**2 + dy**2) ** 0.5

        tagged = dist < (self.tagger.size + self.evader.size)

        scale = max(5, min(20, 50 / (dist + 1)))
        self.tagger.size = scale
        self.evader.size = scale

        return tagged, hit1, hit2

    def get_state(self):
        return [
            self.tagger.x / WIDTH,
            self.tagger.y / HEIGHT,
            self.evader.x / WIDTH,
            self.evader.y / HEIGHT,
        ]


def eval_pair(net_tagger, net_evader):
    score_tagger = 0
    score_evader = 0

    for _ in range(ROUNDS_PER_MATCH):
        game = Game()

        for _ in range(MAX_STEPS):
            state = game.get_state()

            out1 = net_tagger.activate(state)
            out2 = net_evader.activate(state)

            tagged, hit1, hit2 = game.step(out1, out2)

            if hit1:
                score_tagger -= WRAP_PENALTY
            if hit2:
                score_evader -= WRAP_PENALTY

            if tagged:
                score_tagger += TAG_REWARD
                score_evader -= EVADE_PUNISH
                break
        else:
            score_evader += EVADE_REWARD

    return score_tagger, score_evader


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

        # reset fitness
        for _, g in tagger_genomes:
            g.fitness = 0
        for _, g in evader_genomes:
            g.fitness = 0

        # pair randomly
        random.shuffle(evader_genomes)

        for (id1, g1), (id2, g2) in zip(tagger_genomes, evader_genomes):
            net1 = neat.nn.FeedForwardNetwork.create(g1, config)
            net2 = neat.nn.FeedForwardNetwork.create(g2, config)

            s1, s2 = eval_pair(net1, net2)

            g1.fitness += s1
            g2.fitness += s2

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

    # pick best
    best_tagger = max(
        [g for g in pop_tagger.population.values() if g.fitness is not None],
        key=lambda g: g.fitness
    )
    best_evader = max(
        [g for g in pop_evader.population.values() if g.fitness is not None],
        key=lambda g: g.fitness
    )

    return best_tagger, best_evader, config


def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()

        out1 = net1.activate(state)
        out2 = net2.activate(state)

        tagged, _, _ = game.step(out1, out2)

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


if __name__ == '__main__':
    best_tagger, best_evader, config = run('config.txt')

    net1 = neat.nn.FeedForwardNetwork.create(best_tagger, config)
    net2 = neat.nn.FeedForwardNetwork.create(best_evader, config)

    render_game(net1, net2)
