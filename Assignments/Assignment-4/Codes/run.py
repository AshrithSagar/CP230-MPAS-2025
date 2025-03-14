"""
run.py
Hamstrung sqaud game
"""

from hamstrung_squad import HamstrungSquadEnv


def main():
    env = HamstrungSquadEnv(
        size=(10, 10),
        start={"pursuer": (0, 0), "evader": (9, 9)},
        max_steps=10,
        render_mode="ansi",
        seed=42,
    )
    env.reset(seed=42)
    env.render()


if __name__ == "__main__":
    main()
