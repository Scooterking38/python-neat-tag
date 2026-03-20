# ===== main.py =====
import neat
import random
import cv2
import numpy as np

WIDTH, HEIGHT = 200, 200
MAX_STEPS = 300
SPEED = 3

class Square:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.size = 10

    def move(self, dx, dy):
        # toroidal wrapping (folded screen)
        self.x = (self.x + dx) % WIDTH
        self.y = (self.y + dy) % HEIGHT

class Game:
    def __init__(self):
        self.tagger = Square()
        self.evader = Square()
        self.tagger_turn = True
        self.steps = 0

    def step(self, out1, out2):
        def decode(o):
            return (o[0]-0.5)*2*SPEED, (o[1]-0.5)*2*SPEED

        dx1, dy1 = decode(out1)
        dx2, dy2 = decode(out2)

        self.tagger.move(dx1, dy1)
        self.evader.move(dx2, dy2)

        # distance with wrap-around awareness
        dx = min(abs(self.tagger.x - self.evader.x), WIDTH - abs(self.tagger.x - self.evader.x))
        dy = min(abs(self.tagger.y - self.evader.y), HEIGHT - abs(self.tagger.y - self.evader.y))
        dist = (dx**2 + dy**2)**0.5

        tagged = dist < (self.tagger.size + self.evader.size)

        if tagged:
            self.tagger_turn = not self.tagger_turn

        # dynamic scaling
        scale = max(5, min(20, 50/(dist+1)))
        self.tagger.size = scale
        self.evader.size = scale

        self.steps += 1
        return tagged, dist

    def get_state(self):
        return [
            self.tagger.x/WIDTH,
            self.tagger.y/HEIGHT,
            self.evader.x/WIDTH,
            self.evader.y/HEIGHT,
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
        if i+1 >= len(nets): break

        net1 = nets[i]
        net2 = nets[i+1]
        g1 = ge[i]
        g2 = ge[i+1]

        game = Game()

        tagged = False

        for step in range(MAX_STEPS):
            state = game.get_state()
            out1 = net1.activate(state)
            out2 = net2.activate(state)

            tagged, _ = game.step(out1, out2)

            if tagged:
                break

        # ONLY rewards (no punishment)
        if tagged:
            if game.tagger_turn:  # swap already happened, so previous tagger was other
                g2.fitness += 1
            else:
                g1.fitness += 1
        else:
            # evader survived full time
            if game.tagger_turn:
                g2.fitness += 1
            else:
                g1.fitness += 1


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    winner = pop.run(eval_genomes, 30)

    return winner, config


def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()
        out1 = net1.activate(state)
        out2 = net2.activate(state)

        tagged, _ = game.step(out1, out2)

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # draw with wrapping awareness (simple version)
        def draw_square(frame, sq, color):
            x, y, s = int(sq.x), int(sq.y), int(sq.size)
            cv2.rectangle(frame, (x, y), (x+s, y+s), color, -1)

        draw_square(frame, game.tagger, (0,0,255))
        draw_square(frame, game.evader, (255,0,0))

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
