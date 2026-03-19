import pygame

pygame.init()
win = pygame.display.set_mode((600, 400))
def do(genomes, config):
    for _, genome in genomes:
        x, y = 10, 10

        for i in range(100):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
            win.fill((0,0,0))
            inputs = (x/600, y/400)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            output = net.activate(inputs)
            x += round(output[0]) * 20
     
            y += round(output[1]) * 20
            if x > 600:
                x = 10
            if x < 0:
                x = 590
            if y > 400:
                y = 10
            if y < 0:
                y = 10
            pygame.draw.rect(win, (255,0,0), (x, y, 20, 20))
            pygame.display.update()

    pygame.quit()
