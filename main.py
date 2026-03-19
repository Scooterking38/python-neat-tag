import os
import pygame
import neat
import imageio
import time

# ----------- SETTINGS -----------
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
FPS_DELAY = 10  # ms per step
TARGET_POS = (15, 10)  # grid target
MAX_STEPS = 200  # ~3 mins at 10ms per step

# Create folder for frames
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

# ----------- PYGAME HEADLESS -----------
pygame.init()
win = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))

# ----------- EVAL FUNCTION -----------
def eval_genomes(genomes, config):
    frame_counter = 0
    start_time = time.time()

    for _, genome in genomes:
        genome.fitness = 0  # ALWAYS set first

    for _, genome in genomes:
        # Global timeout check
        if time.time() - start_time > 180:
            print("Time limit reached, stopping evaluation early")
            break

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        gx, gy = 0, 0

        for step in range(MAX_STEPS):
            dx = (TARGET_POS[0] - gx) / (GRID_WIDTH - 1)
            dy = (TARGET_POS[1] - gy) / (GRID_HEIGHT - 1)

            output = net.activate((dx, dy))
            direction = output.index(max(output))

            if direction == 0:
                gy -= 1
            elif direction == 1:
                gy += 1
            elif direction == 2:
                gx -= 1
            elif direction == 3:
                gx += 1

            gx = max(0, min(GRID_WIDTH - 1, gx))
            gy = max(0, min(GRID_HEIGHT - 1, gy))

            dist = abs(TARGET_POS[0] - gx) + abs(TARGET_POS[1] - gy)
            genome.fitness += 1 / (dist + 1)

            # Drawing (unchanged)
            win.fill((0, 0, 0))
            pygame.draw.rect(win, (0, 255, 0),
                             (TARGET_POS[0]*CELL_SIZE, TARGET_POS[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(win, (255, 0, 0),
                             (gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))

            frame_path = os.path.join(FRAME_DIR, f"frame_{frame_counter:05d}.png")
            pygame.image.save(win, frame_path)
            frame_counter += 1

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
    pop.run(eval_genomes, 5)  # small generations for demo

    # Build video
    video_path = "simulation.mp4"
    frames = sorted(os.listdir(FRAME_DIR))
    with imageio.get_writer(video_path, fps=30) as video:
        for f in frames:
            video.append_data(imageio.imread(os.path.join(FRAME_DIR, f)))

    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    run()
