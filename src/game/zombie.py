from typing import override

from game.animation import Animation
from game.assets import load_frames
from game.entity import Entity, World
from path_search.breadth import find_path


class Zombie(Entity):
    max_health: int = 1
    speed: float = 2
    damage: int = 1
    detection_range: int = 5

    def __init__(self, col: int, row: int) -> None:
        frames = load_frames("zombieToast.gif", crop=(72, 120, 120, 120))

        self.idle_anim: Animation = Animation(frames[0:2], frame_time=0.3)
        self.move_anim: Animation = Animation(frames[0:8], frame_time=0.08)
        attack_anim = Animation(frames[3:13], frame_time=0.08, loop=False)
        self.death_anim: Animation = Animation(
            frames[16:24], frame_time=0.08, loop=False
        )

        super().__init__(col, row, self.idle_anim, attack_anim)

    @property
    def is_gone(self) -> bool:
        return self.animation is self.death_anim and self.death_anim.finished

    @override
    def take_damage(self, amount: int) -> None:
        super().take_damage(amount)
        if not self.is_alive:
            self.play(self.death_anim)

    @override
    def update(self, dt: float, world: World) -> None:
        super().update(dt, world)

        player = world.player

        if not self.is_alive or self.is_moving or self.is_attacking:
            return

        path = find_path(
            (self.col, self.row), (player.col, player.row), world.is_walkable
        )

        if path is None or len(path) - 1 > self.detection_range:
            self.play(self.idle_anim)
            return

        next_col, next_row = path[1]
        d_col = next_col - self.col
        d_row = next_row - self.row

        if (next_col, next_row) == (player.col, player.row):
            self.face(d_col, d_row)
            self.attack(world)
        elif self.move(d_col, d_row, world):
            self.play(self.move_anim)
        else:
            self.play(self.idle_anim)
