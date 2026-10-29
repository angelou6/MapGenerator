from typing import override

import pygame

from game.animation import Animation
from game.assets import TILE_SIZE, load_frames
from game.entity import Entity, World

DIRECTIONS: dict[int, tuple[int, int]] = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
    pygame.K_w: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_a: (-1, 0),
    pygame.K_d: (1, 0),
}


class Player(Entity):
    max_health: int = 4
    speed: float = 4
    damage: int = 1

    def __init__(self, col: int, row: int) -> None:
        self.run_anim: Animation = Animation(
            load_frames("knightRun.gif"), frame_time=0.1
        )

        attack_frames = load_frames(
            "knightAttack.gif",
            crop=(0, 72, 240, 120),
            size=(TILE_SIZE * 2, TILE_SIZE),
        )
        attack_anim = Animation(attack_frames, frame_time=0.1, loop=False)

        self.coins: int = 0
        self.space_held: bool = False

        super().__init__(col, row, self.run_anim, attack_anim)

    def interact(self, world: World) -> None:
        d_col, d_row = self.facing
        if world.open_chest(self.col + d_col, self.row + d_row):
            self.coins += 1
        else:
            self.attack(world)

    @override
    def update(self, dt: float, world: World) -> None:
        super().update(dt, world)

        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE] and not self.space_held
        self.space_held = keys[pygame.K_SPACE]

        if self.is_attacking:
            return
        self.play(self.run_anim)

        if space_pressed and not self.is_moving:
            self.interact(world)
            return

        for key, (d_col, d_row) in DIRECTIONS.items():
            if keys[key]:
                _ = self.move(d_col, d_row, world)
                break

        if not self.is_moving:
            self.run_anim.reset()
