from pygame.surface import Surface


class Animation:
    def __init__(self, frames: list[Surface], frame_time: float, loop: bool = True):
        self.frames: list[Surface] = frames
        self.frame_time: float = frame_time
        self.loop: bool = loop
        self.time: float = 0

    @property
    def frame(self) -> Surface:
        index = int(self.time / self.frame_time)
        if self.loop:
            index %= len(self.frames)
        else:
            index = min(index, len(self.frames) - 1)
        return self.frames[index]

    @property
    def finished(self) -> bool:
        return not self.loop and self.time >= len(self.frames) * self.frame_time

    def update(self, dt: float) -> None:
        self.time += dt

    def reset(self) -> None:
        self.time = 0
