# ===== main.py =====
import neat
import random
import cv2
import numpy as np

WIDTH, HEIGHT = 600, 600
MAX_STEPS = 500

class Square:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.size = 20

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class Game:
    def __init__(self):
        self.tagger = Square()
        self.evader = Square()
        self.tagger_turn = True

    def step(self, out1, out2):
        speed = 5

        def decode(o):
            return (o[0]-0.5)*2*speed, (o[1]-0.5)*2*speed

        dx1, dy1 = decode(out1)
        dx2, dy2 = decode(out2)

        self.tagger.move(dx1, dy1)
        self.evader.move(dx2, dy2)

        dist = ((self.tagger.x - self.evader.x)**2 + (self.tagger.y - self.evader.y)**2)**0.5

        tagged = dist < (self.tagger.size + self.evader.size)

        if tagged:
            self.tagger_turn = not self.tagger_turn

        # dynamic scaling
        self.tagger.size = max(5, min(50, 100/(dist+1)))
        self.evader.size = self.tagger.size

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

        for step in range(MAX_STEPS):
            state = game.get_state()
            out1 = net1.activate(state)
            out2 = net2.activate(state)

            tagged, dist = game.step(out1, out2)

            if game.tagger_turn:
                g1.fitness += 1/(dist+1)
                g2.fitness += dist
            else:
                g2.fitness += 1/(dist+1)
                g1.fitness += dist


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    winner = pop.run(eval_genomes, 20)

    return winner, config


def render_game(net1, net2):
    game = Game()
    frames = []

    for _ in range(MAX_STEPS):
        state = game.get_state()
        out1 = net1.activate(state)
        out2 = net2.activate(state)
        game.step(out1, out2)

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        cv2.rectangle(frame,
                      (int(game.tagger.x), int(game.tagger.y)),
                      (int(game.tagger.x+game.tagger.size), int(game.tagger.y+game.tagger.size)),
                      (0,0,255), -1)

        cv2.rectangle(frame,
                      (int(game.evader.x), int(game.evader.y)),
                      (int(game.evader.x+game.evader.size), int(game.evader.y+game.evader.size)),
                      (255,0,0), -1)

        frames.append(frame)

    out = cv2.VideoWriter('best.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (WIDTH, HEIGHT))
    for f in frames:
        out.write(f)
    out.release()


if __name__ == '__main__':
    winner, config = run('config.txt')
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # play winner vs itself
    render_game(net, net)
