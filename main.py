import pygame

pygame.init()
win = pygame.display.set_mode((600, 400))
def do(genomes, config):
    x, y = 300, 200

    running = True
    while running:
        pygame.time.delay(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        win.fill((0,0,0))
        inputs = (x/600, y/400)
        output = net.activate(inputs)
        x += output[0] * 10
     
        y += output[1] * 10
        
        pygame.draw.rect(win, (255,0,0), (x, y, 20, 20))
        pygame.display.update()

pygame.quit()
