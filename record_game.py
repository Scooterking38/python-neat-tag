import os
import pickle
import random
import math
import time
import pygame
import neat
import imageio

# -------- SETTINGS --------
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
ROUND_DURATION = 30  # seconds
FPS = 10  # frames per second
VIDEO_OUTPUT = "simulation.mp4"

# -------- HELPER --------
def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def move(pos, direction):
    x, y = pos
    old = (x, y)
    if direction == 0:  # up
        y -= 1
    elif direction == 1:  # down
        y += 1
    elif direction == 2:  # left
        x -= 1
    elif direction == 3:  # right
        x += 1
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return old, True
    return (x, y), False

# -------- DRAW FRAME --------
def draw_frame(surface, chaser_pos, evader_pos):
    surface.fill((0, 0, 0))  # black background
    # draw chaser
    pygame.draw.rect(surface, (255, 0, 0), (chaser_pos[0]*CELL_SIZE, chaser_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # draw evader
    pygame.draw.rect(surface, (0, 255, 0), (evader_pos[0]*CELL_SIZE, evader_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return pygame.surfarray.array3d(surface)

# -------- RECORD ROUND --------
def record_round(chaser_path, evader_path, config_path):
    # Load genomes
    with open(chaser_path, "rb") as f:
        chaser_genome = pickle.load(f)
    with open(evader_path, "rb") as f:
        evader_genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    net_chaser = neat.nn.FeedForwardNetwork.create(chaser_genome, config)
    net_evader = neat.nn.FeedForwardNetwork.create(evader_genome, config)

    pos_chaser = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
    pos_evader = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

    pygame.init()
    surface = pygame.Surface((GRID_WIDTH*CELL_SIZE, GRID_HEIGHT*CELL_SIZE))

    frames = []
    start_time = time.time()
    frame_interval = 1.0 / FPS
    last_frame_time = start_time

    while time.time() - start_time < ROUND_DURATION:
        dx = (pos_evader[0] - pos_chaser[0]) / GRID_WIDTH
        dy = (pos_evader[1] - pos_chaser[1]) / GRID_HEIGHT
        dist = math.sqrt(dx*dx + dy*dy)

        move_chaser = net_chaser.activate((dx, dy, dist)).index(max(net_chaser.activate((dx, dy, dist))))
        move_evader = net_evader.activate((-dx, -dy, dist)).index(max(net_evader.activate((-dx, -dy, dist))))

        pos_chaser, _ = move(pos_chaser, move_chaser)
        pos_evader, _ = move(pos_evader, move_evader)

        # Draw at fixed FPS
        if time.time() - last_frame_time >= frame_interval:
            frames.append(draw_frame(surface, pos_chaser, pos_evader))
            last_frame_time = time.time()

    # Save video
    frames = [frame.swapaxes(0,1) for frame in frames]  # Pygame uses (width,height,3)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=FPS)
    print(f"Video saved as {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_round("best_genomes/chaser_best.pkl", "best_genomes/evader_best.pkl", "config.txt")
