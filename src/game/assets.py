from pathlib import Path

import pygame
from pygame.surface import Surface

TILE_SIZE = 40
# Ruta absoluta para que el juego funcione desde cualquier directorio.
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def load_image(name: str) -> Surface:
    return pygame.image.load(ASSETS_DIR / name).convert_alpha()


def load_frames(
    name: str,
    crop: tuple[int, int, int, int] | None = None,
    size: tuple[int, int] | None = None,
) -> list[Surface]:
    frames: list[Surface] = []
    for frame, _duration in pygame.image.load_animation(ASSETS_DIR / name):
        frame = frame.convert_alpha()
        if crop is not None:
            frame = frame.subsurface(crop)
        frames.append(scale(frame, size))
    return frames


def scale(image: Surface, size: tuple[int, int] | None = None) -> Surface:
    if size is None:
        size = (TILE_SIZE, TILE_SIZE)
    return pygame.transform.scale(image, size)


class Assets:
    def __init__(self) -> None:
        self.tileset: Surface = load_image("tileset.png")
        self.floor: Surface = scale(self.from_tileset(8, 176))
        self.door: Surface = scale(self.from_tileset(40, 208))
        self.chest: Surface = scale(self.from_tileset(0, 224))

        self.rock: Surface = scale(load_image("rock.png"))

    def from_tileset(self, x: int, y: int) -> Surface:
        return self.tileset.subsurface((x, y, 8, 8))
