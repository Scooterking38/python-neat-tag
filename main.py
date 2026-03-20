import neat
import random
import cv2
import numpy as np

WIDTH, HEIGHT = 200, 200
MAX_STEPS = 3000
SPEED = 3

WRAP_PENALTY = 0.2
TAG_REWARD = 5
EVADE_REWARD = 3
EVADE_PUNISH = 3


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
        self.steps = 0

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

        self.steps += 1
        return tagged, hit1, hit2

    def get_state(self):
        return [
            self.tagger.x / WIDTH,
            self.tagger.y / HEIGHT,
            self.evader.x / WIDTH,
            self.evader.y / HEIGHT,
        ]


def eval_genomes(genomes, config):
    nets = []
    ge = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)

    for i in range(0, len(nets), 2):
        if i + 1 >= len(nets):
            break

        net1 = nets[i]
        net2 = nets[i + 1]
        g1 = ge[i]      # tagger
        g2 = ge[i + 1]  # evader

        game = Game()

        for _ in range(MAX_STEPS):
            state = game.get_state()
            out1 = net1.activate(state)
            out2 = net2.activate(state)

            tagged, hit1, hit2 = game.step(out1, out2)

            # wrap penalties
            if hit1:
                g1.fitness -= WRAP_PENALTY
            if hit2:
                g2.fitness -= WRAP_PENALTY

            if tagged:
                # strong reward/punishment
                g1.fitness += TAG_REWARD
                g2.fitness -= EVADE_PUNISH
                break
        else:
            # evader survived full duration
            g2.fitness += EVADE_REWARD


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)

    # increased generations
    winner = pop.run(eval_genomes, 100)

    return winner, config


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
    winner, config = run('config.txt')
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    render_game(net, net)
