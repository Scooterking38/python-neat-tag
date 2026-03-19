import pygame

pygame.init()
win = pygame.display.set_mode((600, 400))

x, y = 300, 200

running = True
while running:
    pygame.time.delay(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    win.fill((0,0,0))
    inputs = (x, y)
    output = net.activate(inputs)
    x += output[0]
    y += output[1]
        
    pygame.draw.rect(win, (255,0,0), (x, y, 20, 20))
    pygame.display.update()

pygame.quit()
