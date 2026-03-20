# ===== main.py =====
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
