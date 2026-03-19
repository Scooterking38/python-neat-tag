import pygame
import neat
import os

# ----------- SETTINGS -----------
CELL_SIZE = 20
GRID_WIDTH = 30   # 600 / 20
GRID_HEIGHT = 20  # 400 / 20
WIN_WIDTH = GRID_WIDTH * CELL_SIZE
WIN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS_DELAY = 50

TARGET_POS = (15, 10)  # grid target

# ----------- PYGAME INIT -----------
pygame.init()
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Grid NEAT AI")

# ----------- EVAL FUNCTION -----------
def eval_genomes(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        # Start at random grid position
        gx = 0
        gy = 0

        for step in range(100):  # steps per genome
            # event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Inputs: relative position to target, normalized
            dx = (TARGET_POS[0] - gx) / (GRID_WIDTH - 1)
            dy = (TARGET_POS[1] - gy) / (GRID_HEIGHT - 1)
            inputs = (dx, dy)

            # Network outputs 4 values for directions
            output = net.activate(inputs)
            direction = output.index(max(output))

            # Move one cell in the chosen direction
            if direction == 0:  # up
                gy -= 1
            elif direction == 1:  # down
                gy += 1
            elif direction == 2:  # left
                gx -= 1
            elif direction == 3:  # right
                gx += 1

            # Clamp inside grid
            gx = max(0, min(GRID_WIDTH - 1, gx))
            gy = max(0, min(GRID_HEIGHT - 1, gy))

            # Fitness = closer to target = higher
            dist = abs(TARGET_POS[0] - gx) + abs(TARGET_POS[1] - gy)
            genome.fitness += 1 / (dist + 1)  # +1 to avoid division by 0

            # DRAW
            win.fill((0, 0, 0))
            pygame.draw.rect(win, (0, 255, 0), (TARGET_POS[0]*CELL_SIZE, TARGET_POS[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(win, (255, 0, 0), (gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.display.update()
            pygame.time.delay(FPS_DELAY)

# ----------- RUN FUNCTION -----------
def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.run(eval_genomes, 20)  # run 20 generations

run()
