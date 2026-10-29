from game.game import Game
from generator.map_utils import generate_random_map


def main():
    mapa = generate_random_map().tolist()
    Game(mapa).run()
