import NEAT

from .player import Player
from .settings import settings
from .simulator import simulate


def main() -> None:

    NEAT.run(
        PlayerClass=Player,
        settings=settings,
        simulate=simulate,
    )


if __name__ == '__main__':
    main()