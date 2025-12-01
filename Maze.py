import pygame
import random
import heapq

# Maze settings
W, H = 51, 31
CELL = 10
UI_HEIGHT = 60

pygame.init()
screen = pygame.display.set_mode((W * CELL, H * CELL + UI_HEIGHT))
pygame.display.set_caption("Maze Generator Solver")
font = pygame.font.SysFont(None, 28)

# Maze
maze = [[1 for _ in range(W)] for _ in range(H)]

def neighbors(x, y):
    dirs = [(2,0), (-2,0), (0,2), (0,-2)]
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            yield nx, ny, dx, dy

# Prim's algorithm
sx, sy = 1, 1
maze[sy][sx] = 0
walls = []

for nx, ny, dx, dy in neighbors(sx, sy):
    walls.append((nx, ny, dx, dy))

while walls:
    nx, ny, dx, dy = random.choice(walls)
    walls.remove((nx, ny, dx, dy))
    if maze[ny][nx] == 1:
        mx, my = nx - dx//2, ny - dy//2
        maze[my][mx] = 0
        maze[ny][nx] = 0
        for fx, fy, fdx, fdy in neighbors(nx, ny):
            if maze[fy][fx] == 1:
                walls.append((fx, fy, fdx, fdy))

# Borders
for x in range(W):
    maze[0][x] = maze[H-1][x] = 1
for y in range(H):
    maze[y][0] = maze[y][W-1] = 1

# Randomize start & goal
def randomize_positions():
    global start_x, start_y, goal_x, goal_y, path
    path = []
    path_cells = [(x, y) for y in range(1, H-1) for x in range(1, W-1) if maze[y][x] == 0]
    start_x, start_y = random.choice(path_cells)
    path_cells.remove((start_x, start_y))
    goal_x, goal_y = random.choice(path_cells)

randomize_positions()

# Buttons
button_randomize = pygame.Rect(10, H*CELL + 10, 160, 40)  # bottom-left
button_a_star   = pygame.Rect(W*CELL - 170, H*CELL + 10, 160, 40)  # bottom-right

# A* algorithm
def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and maze[ny][nx] == 0:
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return []

path = []

# Helper to draw styled button
def draw_button(rect, text, base_color, hover_color):
    mouse_pos = pygame.mouse.get_pos()
    color = hover_color if rect.collidepoint(mouse_pos) else base_color
    pygame.draw.rect(screen, color, rect, border_radius=10)
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_randomize.collidepoint(event.pos):
                randomize_positions()
            elif button_a_star.collidepoint(event.pos):
                path = astar((start_x, start_y), (goal_x, goal_y))

    # Draw maze
    for y in range(H):
        for x in range(W):
            if (x, y) == (start_x, start_y):
                color = (0, 180, 0)  # green start
            elif (x, y) == (goal_x, goal_y):
                color = (255, 100, 0)  # orange goal
            elif (x, y) in path:
                color = (250, 250, 0)  # gold path
            else:
                color = (255, 255, 255) if maze[y][x] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (x*CELL, y*CELL, CELL, CELL))

    # Draw buttons with hover effect
    draw_button(button_randomize, "Randomize", (0, 120, 255), (0, 180, 255))
    draw_button(button_a_star, "Run A*", (200, 0, 0), (255, 50, 50))

    pygame.display.flip()

pygame.quit()
