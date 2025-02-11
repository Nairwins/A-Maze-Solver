import pygame
import sys
import random
import heapq
from m import binary_matrix
# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
i = 0
def cell(x , y):
    if isinstance(x, list):
        return 50 if len(x)>10 else 200
    else :
        if x.all() == y.all(): return 3
        else : return 50


# Colors
BLACK = (0, 0, 0)  # Wall color
WHITE = (250, 250, 235)  # Path color
YELLOW = (255, 255, 0)  # Target color
DARK_GREEN = (0, 100, 0)  # Star color
BLUE = (0, 0, 255)  # Button color
RED = (255, 0, 0)  # Red button color
GREEN = (0, 255, 0)  # Trail color

# Define the maze layout
# 0 represents a path, 1 represents a wall
def MZ(input,Y):
    if input == 1: return [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    elif input == 2 : return [[0,0,0],
                   [0,1,1],
                   [0,0,0]]
    else : return Y

maze = MZ(i,binary_matrix)

CELL_SIZE = cell(maze , binary_matrix)


# Calculate grid dimensions
GRID_WIDTH = len(maze[0])
GRID_HEIGHT = len(maze)

# Set up the screen
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Maze")

# Randomized positions
def get_random_path_position():
    while True:
        row = random.randint(0, GRID_HEIGHT - 1)
        col = random.randint(0, GRID_WIDTH - 1)
        if maze[row][col] == 0:  # Ensure it's on a path
            return row, col

target_pos = get_random_path_position()
star_pos = get_random_path_position()

while target_pos == star_pos:
    star_pos = get_random_path_position()

# Button rectangles for red and blue buttons
red_button_rect = pygame.Rect(10, GRID_HEIGHT * CELL_SIZE - 37, 150, 35)
blue_button_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE - 160, GRID_HEIGHT * CELL_SIZE - 37, 150, 35)

def draw_maze():
    """Draws the maze on the screen."""
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            color = WHITE if maze[row][col] == 0 else BLACK
            pygame.draw.rect(
                screen,
                color,
                (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

    # Draw target
    pygame.draw.rect(
        screen,
        YELLOW,
        (target_pos[1] * CELL_SIZE, target_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Draw star
    pygame.draw.rect(
        screen,
        DARK_GREEN,
        (star_pos[1] * CELL_SIZE, star_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

def draw_button():
    # Red button
    pygame.draw.rect(screen, RED, red_button_rect)
    font = pygame.font.SysFont(None, 24)
    text = font.render("Start Pathfinding", True, WHITE)
    screen.blit(text, (red_button_rect.x + 10, red_button_rect.y + 10))

    # Blue button
    pygame.draw.rect(screen, BLUE, blue_button_rect)
    text = font.render("Randomize", True, WHITE)
    screen.blit(text, (blue_button_rect.x + 10, blue_button_rect.y + 10))

# A* Pathfinding Algorithm
def astar(start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in [(current[0] - 1, current[1]), (current[0] + 1, current[1]), (current[0], current[1] - 1), (current[0], current[1] + 1)]:
            if 0 <= neighbor[0] < GRID_HEIGHT and 0 <= neighbor[1] < GRID_WIDTH and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = current_g + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

    return []

def main():
    global target_pos, star_pos
    clock = pygame.time.Clock()
    path = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if red_button_rect.collidepoint(event.pos):
                    path = astar(star_pos, target_pos)
                elif blue_button_rect.collidepoint(event.pos):
                    target_pos = get_random_path_position()
                    star_pos = get_random_path_position()
                    path = []  # Clear the path when randomizing positions

        # Draw the maze and buttons
        screen.fill(BLACK)
        draw_maze()
        draw_button()

        # Draw the path if it exists
        for idx, pos in enumerate(path):
            # If it's the last position in the path, make it red (target reached)
            color = RED if idx == len(path) - 1 else GREEN
            pygame.draw.rect(
                screen,
                color,
                (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

        # Update the screen
        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
