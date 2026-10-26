import pygame
from pygame.font import Font
from pygame.surface import Surface

from game.assets import TILE_SIZE, load_image
from game.player import Player


class Hud:
    height: int = TILE_SIZE
    margin: int = 8

    def __init__(self) -> None:
        sheet = load_image("hudElements.png")
        self.heart: Surface = sheet.subsurface((0, 0, 32, 32))
        self.coin: Surface = sheet.subsurface((32, 0, 32, 32))

        self.font: Font = pygame.font.Font(None, 32)

    def draw(self, screen: Surface, player: Player, top: int) -> None:
        bar = pygame.Rect(0, top, screen.get_width(), self.height)
        _ = screen.fill("black", bar)

        x = self.margin
        x = self.draw_counter(screen, self.heart, player.health, x, top)
        x += self.margin * 3
        _ = self.draw_counter(screen, self.coin, player.coins, x, top)

    def draw_counter(
        self, screen: Surface, icon: Surface, value: int, x: int, top: int
    ) -> int:
        icon_y = top + (self.height - icon.get_height()) // 2
        _ = screen.blit(icon, (x, icon_y))
        x += icon.get_width() + self.margin

        text = self.font.render(f"x {value}", False, "white")
        text_y = top + (self.height - text.get_height()) // 2
        _ = screen.blit(text, (x, text_y))
        return x + text.get_width()
