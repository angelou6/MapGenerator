from typing import Protocol

import pygame
from pygame.surface import Surface

from game.animation import Animation
from game.assets import TILE_SIZE


class World(Protocol):
    @property
    def player(self) -> "Entity": ...

    def is_walkable(self, col: int, row: int) -> bool: ...

    def is_free(self, col: int, row: int) -> bool: ...

    def entity_at(self, col: int, row: int) -> "Entity | None": ...

    def open_chest(self, col: int, row: int) -> bool: ...


class Entity:
    max_health: int = 1
    speed: float = 4
    damage: int = 1

    def __init__(
        self, col: int, row: int, animation: Animation, attack_anim: Animation
    ):
        self.col: int = col
        self.row: int = row

        self.x: float = col
        self.y: float = row

        self.facing: tuple[int, int] = (1, 0)
        self.facing_left: bool = False

        self.animation: Animation = animation
        self.attack_anim: Animation = attack_anim
        self.health: int = self.max_health

    @property
    def is_moving(self) -> bool:
        return (self.x, self.y) != (self.col, self.row)

    @property
    def is_attacking(self) -> bool:
        return self.animation is self.attack_anim and not self.attack_anim.finished

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def take_damage(self, amount: int):
        self.health = max(0, self.health - amount)

    def move(self, d_col: int, d_row: int, world: World) -> bool:
        if self.is_moving:
            return False

        self.face(d_col, d_row)

        target_col = self.col + d_col
        target_row = self.row + d_row
        if not world.is_free(target_col, target_row):
            return False

        self.col = target_col
        self.row = target_row
        return True

    def face(self, d_col: int, d_row: int) -> None:
        self.facing = (d_col, d_row)
        if d_col != 0:
            self.facing_left = d_col < 0

    def attack(self, world: World) -> None:
        self.animation = self.attack_anim
        self.attack_anim.reset()

        d_col, d_row = self.facing
        target = world.entity_at(self.col + d_col, self.row + d_row)
        if target is not None:
            target.take_damage(self.damage)

    def update(self, dt: float, world: World):  # pyright: ignore[reportUnusedParameter]
        step = self.speed * dt
        self.x = approach(self.x, self.col, step)
        self.y = approach(self.y, self.row, step)
        self.animation.update(dt)

    def play(self, animation: Animation) -> None:
        if self.animation is not animation:
            self.animation = animation
            self.animation.reset()

    def draw(self, screen: Surface):
        frame = self.animation.frame
        x = self.x * TILE_SIZE
        y = self.y * TILE_SIZE

        if self.facing_left:
            frame = pygame.transform.flip(frame, True, False)
            x -= frame.get_width() - TILE_SIZE

        y -= frame.get_height() - TILE_SIZE
        _ = screen.blit(frame, (x, y))


def approach(value: float, target: float, step: float) -> float:
    if value < target:
        return min(value + step, target)
    return max(value - step, target)
