"""
utils/__init__.py
"""

from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="utils.grid_world:GridWorld",
    kwargs={
        "size": (40, 40),
        "start": (0, 0),
        "goal": [(38, 38), (38, 39), (39, 38), (39, 39)],
        "obstacles": [
            [(8, 8), (8, 9), (9, 8), (9, 9), (10, 8)],
            [(18, 18), (18, 19), (19, 18), (19, 19), (20, 18), (19, 17)],
            [(25, 25), (25, 26), (26, 25), (26, 26), (27, 25)],
            [(30, 10), (30, 11), (31, 10), (31, 11), (32, 10)],
            [(35, 35), (35, 36), (36, 35), (36, 36), (37, 35)],
        ],
    },
)
