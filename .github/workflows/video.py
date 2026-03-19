name: Generate NEAT Simulation Video

on:
  workflow_dispatch:  # manual trigger
  push:
    paths:
      - "best_genomes/**"  # triggers when any file in the folder changes

jobs:
  generate-video:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pygame neat-python imageio[ffmpeg]

      # Headless Pygame
      - name: Set SDL to dummy (headless mode)
        run: echo "SDL_VIDEODRIVER=dummy" >> $GITHUB_ENV

      - name: Generate video
        run: python record_game.py

      - name: Upload simulation video
        uses: actions/upload-artifact@v4
        with:
          name: simulation-video
          path: simulation.mp4
