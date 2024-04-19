import NEAT

from .player import Player
from .settings import settings
from .simulator import simulate


def main() -> None:

    NEAT.run(
        PlayerClass=Player,
        simulate=simulate,
        settings=settings,
    )


if __name__ == '__main__':
    main()