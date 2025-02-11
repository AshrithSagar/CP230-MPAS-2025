import numpy as np

def create_grid():
    grid_size = 40
    grid = np.full((grid_size, grid_size), '.')
    
    # Define start position
    start = (0, 0)
    grid[start] = 'S'
    
    # Define goal area (4 grids together along the diagonal)
    goal_area = [(38, 38), (38, 39), (39, 38), (39, 39)]
    for goal in goal_area:
        grid[goal] = 'G'
    
    # Define obstacles (bunch of 4-6 grids with different shapes)
    obstacles = [
        [(8, 8), (8, 9), (9, 8), (9, 9), (10, 8)],  # Square with an extension
        [(18, 18), (18, 19), (19, 18), (19, 19), (20, 18), (19, 17)],  # L-shape
        [(25, 25), (25, 26), (26, 25), (26, 26), (27, 25)],  # Square with an extension
        [(30, 10), (30, 11), (31, 10), (31, 11), (32, 10)],  # Rectangular block
        [(35, 35), (35, 36), (36, 35), (36, 36), (37, 35)]   # Square with extension
    ]
    
    for obs_group in obstacles:
        for obs in obs_group:
            grid[obs] = 'X'
    
    return grid, start, goal_area, obstacles

def save_grid_to_file(grid, filename="grid.txt"):
    with open(filename, "w") as f:
        for row in grid:
            f.write(" ".join(row) + "\n")

grid, start, goal_area, obstacles = create_grid()
save_grid_to_file(grid)
