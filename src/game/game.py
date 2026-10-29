import pygame
from pygame.surface import Surface
from pygame.time import Clock

from game.assets import TILE_SIZE, Assets
from game.entity import Entity
from game.hud import Hud
from game.player import Player
from game.zombie import Zombie
from representation import Elemento


class Game:
    map_size: int = 15 * TILE_SIZE
    window_size: tuple[int, int] = (map_size, map_size + Hud.height)

    def __init__(self, mapa: list[list[Elemento]]) -> None:
        _ = pygame.init()
        pygame.display.set_caption("Juego Prueba")
        self.clock: Clock = pygame.time.Clock()
        self.screen: Surface = pygame.display.set_mode(self.window_size)

        self.initial_mapa: list[list[Elemento]] = [list(line) for line in mapa]
        self.assets: Assets = Assets()
        self.hud: Hud = Hud()

        self.sprites: dict[Elemento, Surface] = {
            Elemento.OBSTACULO: self.assets.rock,
            Elemento.RECOMPENZA: self.assets.chest,
            Elemento.META: self.assets.door,
        }

        self.mapa: list[list[Elemento]] = []
        self.player: Player = Player(0, 0)
        self.zombies: list[Zombie] = []
        self.reset()

    def reset(self) -> None:
        self.mapa = [list(line) for line in self.initial_mapa]

        self.player = Player(0, 0)
        self.zombies = []
        for row, line in enumerate(self.mapa):
            for col, element in enumerate(line):
                if element == Elemento.ENEMIGO:
                    self.zombies.append(Zombie(col, row))
                    self.mapa[row][col] = Elemento.TRANSITABLE

    @property
    def entities(self) -> list[Entity]:
        return [*self.zombies, self.player]

    def entity_at(self, col: int, row: int) -> Entity | None:
        for entity in self.entities:
            if (entity.col, entity.row) == (col, row):
                return entity
        return None

    def in_bounds(self, col: int, row: int) -> bool:
        size = len(self.mapa)
        return 0 <= col < size and 0 <= row < size

    def is_walkable(self, col: int, row: int) -> bool:
        if not self.in_bounds(col, row):
            return False
        return self.mapa[row][col] not in (Elemento.OBSTACULO, Elemento.RECOMPENZA)

    def is_free(self, col: int, row: int) -> bool:
        return self.is_walkable(col, row) and self.entity_at(col, row) is None

    def open_chest(self, col: int, row: int) -> bool:
        if not self.in_bounds(col, row) or self.mapa[row][col] != Elemento.RECOMPENZA:
            return False

        self.mapa[row][col] = Elemento.TRANSITABLE
        return True

    def draw_map(self):
        for row, line in enumerate(self.mapa):
            for col, element in enumerate(line):
                position = (col * TILE_SIZE, row * TILE_SIZE)
                _ = self.screen.blit(self.assets.floor, position)

                if element in self.sprites:
                    _ = self.screen.blit(self.sprites[element], position)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            dt = self.clock.tick(60) / 1000
            for entity in self.entities:
                entity.update(dt, self)

            self.zombies = [z for z in self.zombies if not z.is_gone]

            if not self.player.is_alive:
                self.reset()

            self.draw_map()
            for entity in self.entities:
                entity.draw(self.screen)
            self.hud.draw(self.screen, self.player, top=self.map_size)
            pygame.display.flip()

        pygame.quit()
