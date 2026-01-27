import pygame
import random
import sys
import heapq
import math
import time
import numpy as np

try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    GPU_AVAILABLE = False


# --- Configuration ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
UI_TOP_HEIGHT = 100
UI_BOT_HEIGHT = 100

# Color Palette
NV_GREEN = (118, 185, 0)
GREEN = (0, 255, 0)
RED = (200, 250, 0)           # Large Maze Path
DARK_GREEN = (0, 100, 0)      # Goal and Roads (Search)
ORRANGE = (255, 100, 0)       # Path
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)       # Preview Walls
YELLOW = (255, 255, 0)        # Start Point

pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Lab")
FONT_S = pygame.font.SysFont("Consolas", 14, bold=True)
FONT_M = pygame.font.SysFont("Consolas", 20, bold=True)
FONT_L = pygame.font.SysFont("Consolas", 36, bold=True)

# ---------------- GPU KERNEL for wavefront expand ----------------
if GPU_AVAILABLE:
    @cuda.jit
    def gpu_expand_kernel(walls_flat, W, H, frontier, dist, new_frontier, step):
        idx = cuda.grid(1)
        total = W * H
        if idx >= total:
            return
        if frontier[idx] == 0:
            return
        x = idx // H
        y = idx % H

        base = idx * 4
        
        if walls_flat[base + 0] == 0 and y - 1 >= 0:
            nb_idx = x * H + (y - 1)
            old = cuda.atomic.compare_and_swap(dist[nb_idx:], -1, step + 1)
            if old == -1:
                new_frontier[nb_idx] = 1
                
        if walls_flat[base + 1] == 0 and x + 1 < W:
            nb_idx = (x + 1) * H + y
            old = cuda.atomic.compare_and_swap(dist[nb_idx:], -1, step + 1)
            if old == -1:
                new_frontier[nb_idx] = 1
                
        if walls_flat[base + 2] == 0 and y + 1 < H:
            nb_idx = x * H + (y + 1)
            old = cuda.atomic.compare_and_swap(dist[nb_idx:], -1, step + 1)
            if old == -1:
                new_frontier[nb_idx] = 1
                
        if walls_flat[base + 3] == 0 and x - 1 >= 0:
            nb_idx = (x - 1) * H + y
            old = cuda.atomic.compare_and_swap(dist[nb_idx:], -1, step + 1)
            if old == -1:
                new_frontier[nb_idx] = 1


class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.visited = False
        self.walls = [True, True, True, True] # T, R, B, L

def generate_static_maze(width, height):
    cells = [[Cell(x, y) for y in range(height)] for x in range(width)]
    walls = []
    cells[0][0].visited = True
    def add_walls(x, y):
        for i, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not cells[nx][ny].visited:
                walls.append((x, y, nx, ny, i))
    add_walls(0, 0)
    while walls:
        x1, y1, x2, y2, d = walls.pop(random.randrange(len(walls)))
        if not cells[x2][y2].visited:
            cells[x1][y1].walls[d] = False
            cells[x2][y2].walls[(d+2)%4] = False
            cells[x2][y2].visited = True
            add_walls(x2, y2)
    return cells

class MazeApp:
    def __init__(self):
        self.state = "MENU"
        self.grid_x, self.grid_y = 20, 20
        self.input_x, self.input_y = "20", "20"
        self.active_input = None
        self.cursor_timer = 0
        
        # Zoom & Pan State
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_mouse_pos = (0,0)
        
        self.visual_gen = True

        # compute_mode selector
        self.compute_mode = "CPU_ASTAR"
        self.gpu_available = GPU_AVAILABLE

        # Path visualisation toggle
        self.path_visual = True
        
        # Previews WHITE walls
        self.previews = {
            (5, 5): generate_static_maze(5, 5),
            (20, 5): generate_static_maze(20, 5),
            (30, 30): generate_static_maze(30, 30),
            (70, 50): generate_static_maze(70, 50)
        }
        
        self.cells = []
        self.walls_list = []
        self.generating = False
        self.solving = False
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.path, self.searched = [], set()

        # compute-accumulators
        self.solve_compute_time = 0.0
        self.last_solve_time = None
        self.last_solve_method = None

        # flood state
        self._flood_state = None

    def init_maze(self, x, y):
        self.grid_x, self.grid_y = x, y
        self.cells = [[Cell(ix, iy) for iy in range(y)] for ix in range(x)]
        self.walls_list = []
        self.path, self.searched, self.solving = [], set(), False
        self.zoom = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.state = "MAZE"
        
        rx, ry = random.randrange(x), random.randrange(y)
        self.cells[rx][ry].visited = True
        self.add_walls_to_list(rx, ry)
        
        if not self.visual_gen:
            while self.walls_list: self.prim_step()
            self.generating = False
        else:
            self.generating = True

        self.randomize_points()

    def add_walls_to_list(self, x, y):
        for i, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                if not self.cells[nx][ny].visited:
                    self.walls_list.append((x, y, nx, ny, i))

    def prim_step(self):
        if not self.walls_list: self.generating = False; return
        idx = random.randrange(len(self.walls_list))
        x1, y1, x2, y2, d = self.walls_list.pop(idx)
        if not self.cells[x2][y2].visited:
            self.cells[x1][y1].walls[d] = False
            self.cells[x2][y2].walls[(d+2)%4] = False
            self.cells[x2][y2].visited = True
            self.add_walls_to_list(x2, y2)

    def randomize_points(self):
        self.start_pos = (random.randrange(self.grid_x), random.randrange(self.grid_y))
        self.goal_pos = (random.randrange(self.grid_x), random.randrange(self.grid_y))
        while self.goal_pos == self.start_pos:
            self.goal_pos = (random.randrange(self.grid_x), random.randrange(self.grid_y))
        self.path, self.searched, self.solving = [], set(), False

    # ---------------- A* start (original behavior preserved) ----------------
    def start_astar(self):
        self.solve_compute_time = 0.0
        self.last_solve_time = None
        self.last_solve_method = None

        if self.compute_mode == "GPU_FLOOD":
            self.last_solve_method = "GPU Flood"
            if self.path_visual and self.gpu_available:
                ok = self._init_flood_state_gpu()
                if ok:
                    self.solving = True
                    return
                else:
                    start = time.perf_counter()
                    found = self.compute_floodfill_path(use_gpu=False)
                    self.last_solve_time = time.perf_counter() - start
                    return
            else:
                start = time.perf_counter()
                found = self.compute_floodfill_path(use_gpu=(self.gpu_available and self.compute_mode=="GPU_FLOOD"))
                self.last_solve_time = time.perf_counter() - start
                return

        self.path, self.searched = [], set()
        self.open_set = [(0, self.start_pos)]
        self.came_from = {}
        self.g_score = {(x, y): float('inf') for x in range(self.grid_x) for y in range(self.grid_y)}
        self.g_score[self.start_pos] = 0

        self.last_solve_method = "CPU A*"
        if not self.path_visual:
            while self.open_set:
                if not self._timed_astar_step(): break
            self.last_solve_time = self.solve_compute_time
        else:
            self.solving = True

    def _timed_astar_step(self):
        t0 = time.perf_counter()
        res = self.astar_step()
        t1 = time.perf_counter()
        self.solve_compute_time += (t1 - t0)
        return res

    def astar_step(self):
        if not self.open_set: self.solving = False; return False
        curr = heapq.heappop(self.open_set)[1]
        self.searched.add(curr)
        if curr == self.goal_pos:
            self.path = [curr]
            while curr in self.came_from:
                curr = self.came_from[curr]
                self.path.append(curr)
            self.last_solve_time = self.solve_compute_time
            self.solving = False
            return False
        x, y = curr
        for i, (dx, dy) in enumerate([(0,-1), (1,0), (0,1), (-1,0)]):
            if not self.cells[x][y].walls[i]:
                nb = (x+dx, y+dy)
                tg = self.g_score[curr] + 1
                if tg < self.g_score[nb]:
                    self.came_from[nb] = curr
                    self.g_score[nb] = tg
                    f = tg + abs(nb[0]-self.goal_pos[0]) + abs(nb[1]-self.goal_pos[1])
                    heapq.heappush(self.open_set, (f, nb))
        return True

    def _init_flood_state_gpu(self):
        try:
            W, H = self.grid_x, self.grid_y
            total = W * H
            walls_flat = np.zeros((W * H * 4), dtype=np.int8)
            for x in range(W):
                for y in range(H):
                    base = (x * H + y) * 4
                    walls_flat[base + 0] = 1 if self.cells[x][y].walls[0] else 0
                    walls_flat[base + 1] = 1 if self.cells[x][y].walls[1] else 0
                    walls_flat[base + 2] = 1 if self.cells[x][y].walls[2] else 0
                    walls_flat[base + 3] = 1 if self.cells[x][y].walls[3] else 0

            dist = np.full(total, -1, dtype=np.int32)
            frontier = np.zeros(total, dtype=np.int8)
            new_frontier = np.zeros(total, dtype=np.int8)

            sx, sy = self.start_pos
            gx, gy = self.goal_pos
            start_idx = sx * H + sy
            goal_idx = gx * H + gy
            dist[start_idx] = 0
            frontier[start_idx] = 1

            d_walls = cuda.to_device(walls_flat)
            d_dist = cuda.to_device(dist)
            d_frontier = cuda.to_device(frontier)
            d_new_frontier = cuda.to_device(new_frontier)

            threads_per_block = 256
            blocks = (total + threads_per_block - 1) // threads_per_block

            self._flood_state = {
                "W": W, "H": H, "total": total,
                "d_walls": d_walls, "d_dist": d_dist,
                "d_frontier": d_frontier, "d_new_frontier": d_new_frontier,
                "dist_host": dist, "new_frontier_host": new_frontier,
                "start_idx": start_idx, "goal_idx": goal_idx,
                "blocks": blocks, "threads": threads_per_block,
                "step": 0, "reached": False
            }
            self.solve_compute_time = 0.0
            self.last_solve_time = None
            self.last_solve_method = "GPU Flood"
            return True
        except Exception:
            self._flood_state = None
            return False

    def _step_flood_gpu(self):
        st = self._flood_state
        if st is None: return False
        
        t0 = time.perf_counter()
        st["d_new_frontier"].copy_to_device(np.zeros_like(st["new_frontier_host"]))
        gpu_expand_kernel[st["blocks"], st["threads"]](
            st["d_walls"], st["W"], st["H"],
            st["d_frontier"], st["d_dist"], st["d_new_frontier"], st["step"]
        )
        st["d_new_frontier"].copy_to_host(st["new_frontier_host"])
        st["d_dist"].copy_to_host(st["dist_host"])
        t1 = time.perf_counter()
        self.solve_compute_time += (t1 - t0)

        if st["dist_host"][st["goal_idx"]] != -1:
            st["reached"] = True
            H = st["H"]
            dist = st["dist_host"]
            curidx = st["goal_idx"]
            curdist = int(dist[curidx])
            cx, cy = curidx // H, curidx % H
            path = [(cx, cy)]
            while curdist > 0:
                x, y = path[-1]
                found = False
                for i, (dx, dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
                    if not self.cells[x][y].walls[i]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < st["W"] and 0 <= ny < st["H"]:
                            nidx = nx * H + ny
                            if dist[nidx] == curdist - 1:
                                path.append((nx, ny))
                                curdist -= 1
                                found = True
                                break
                if not found: return False
            path.reverse()
            self.path = path
            self.searched = set()
            for i_val in range(st["total"]):
                if dist[i_val] != -1:
                    self.searched.add((i_val // H, i_val % H))
            self.last_solve_time = self.solve_compute_time
            self.solving = False
            return True

        st["d_frontier"], st["d_new_frontier"] = st["d_new_frontier"], st["d_frontier"]
        st["new_frontier_host"].fill(0)
        st["step"] += 1
        self.searched = set()
        for i_val in range(st["total"]):
            if st["dist_host"][i_val] != -1:
                self.searched.add((i_val // st["H"], i_val % st["H"]))
        return True

    def compute_floodfill_path(self, use_gpu=True):
        W, H = self.grid_x, self.grid_y

        def cpu_bfs():
            q = [self.start_pos]
            parent = {self.start_pos: None}
            head = 0
            while head < len(q):
                cur = q[head]; head += 1
                if cur == self.goal_pos: break
                x, y = cur
                for i, (dx, dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
                    if not self.cells[x][y].walls[i]:
                        nb = (x+dx, y+dy)
                        if nb not in parent:
                            parent[nb] = cur
                            q.append(nb)
            if self.goal_pos not in parent: return False
            path = [self.goal_pos]
            cur = self.goal_pos
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            self.path, self.searched = path, set(parent.keys())
            return True

        if use_gpu and GPU_AVAILABLE:
            try:
                walls_flat = np.zeros((W * H * 4), dtype=np.int8)
                for x in range(W):
                    for y in range(H):
                        base = (x * H + y) * 4
                        for i in range(4): walls_flat[base+i] = 1 if self.cells[x][y].walls[i] else 0

                total = W * H
                dist, frontier, new_frontier = np.full(total, -1, dtype=np.int32), np.zeros(total, dtype=np.int8), np.zeros(total, dtype=np.int8)
                sx, sy = self.start_pos
                gx, gy = self.goal_pos
                start_idx, goal_idx = sx * H + sy, gx * H + gy
                dist[start_idx], frontier[start_idx] = 0, 1

                d_walls, d_dist, d_frontier, d_new_frontier = cuda.to_device(walls_flat), cuda.to_device(dist), cuda.to_device(frontier), cuda.to_device(new_frontier)
                threads = 256
                blocks = (total + threads - 1) // threads

                step = 0
                while True:
                    d_new_frontier.copy_to_device(np.zeros_like(new_frontier))
                    gpu_expand_kernel[blocks, threads](d_walls, W, H, d_frontier, d_dist, d_new_frontier, step)
                    d_dist.copy_to_host(dist)
                    d_new_frontier.copy_to_host(new_frontier)
                    step += 1
                    if dist[goal_idx] != -1: break
                    if np.count_nonzero(new_frontier) == 0: break
                    d_frontier, d_new_frontier = d_new_frontier, d_frontier

                if dist[goal_idx] == -1: return cpu_bfs()
                path, curidx, curdist = [], goal_idx, int(dist[goal_idx])
                path.append((curidx // H, curidx % H))
                while curdist > 0:
                    x, y = path[-1]
                    for i, (dx, dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
                        if not self.cells[x][y].walls[i]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < W and 0 <= ny < H and dist[nx*H+ny] == curdist - 1:
                                path.append((nx, ny)); curdist -= 1; break
                path.reverse()
                self.path = path
                self.searched = { (i//H, i%H) for i, d in enumerate(dist) if d != -1 }
                return True
            except Exception: return cpu_bfs()
        return cpu_bfs()

    def draw_checkbox(self, x, y, label, state, mouse):
        rect = pygame.Rect(x, y, 22, 22)
        hover = rect.collidepoint(mouse)
        color = NV_GREEN if hover or state else WHITE
        pygame.draw.rect(SCREEN, color, rect, 2)
        if state: pygame.draw.rect(SCREEN, color, rect.inflate(-8, -8))
        SCREEN.blit(FONT_S.render(label, True, WHITE), (rect.right + 8, rect.y + 4))
        return rect

    def draw_nv_btn(self, rect, text, color, icon=None):
        mouse = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse)
        draw_color = color if hover else WHITE
        pygame.draw.rect(SCREEN, (30,30,30), rect)
        pygame.draw.rect(SCREEN, draw_color, rect, 2)
        if hover: pygame.draw.rect(SCREEN, draw_color, rect.inflate(4, 4), 1)
        if icon == "back":
            pygame.draw.lines(SCREEN, draw_color, False, [(rect.x+20, rect.centery-6), (rect.x+10, rect.centery), (rect.x+20, rect.centery+6)], 3)
        t = FONT_M.render(text, True, draw_color)
        SCREEN.blit(t, t.get_rect(center=rect.center))

    def draw_mini_maze(self, rect, cells, w_count, h_count):
        pygame.draw.rect(SCREEN, BLACK, rect)
        cw, ch = rect.width / w_count, rect.height / h_count
        for x in range(w_count):
            for y in range(h_count):
                px, py = rect.x + x*cw, rect.y + y*ch
                w = cells[x][y].walls
                if w[0]: pygame.draw.line(SCREEN, WHITE, (px, py), (px+cw, py), 1)
                if w[3]: pygame.draw.line(SCREEN, WHITE, (px, py), (px, py+ch), 1)
        pygame.draw.rect(SCREEN, WHITE, rect, 1)

    def run(self):
        clock = pygame.time.Clock()
        while True:
            SCREEN.fill(BLACK)
            self.cursor_timer = (self.cursor_timer + 1) % 60
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_clicks(mouse)
                    if self.state == "MAZE" and event.button == 1:
                        if UI_TOP_HEIGHT < mouse[1] < SCREEN_HEIGHT - UI_BOT_HEIGHT:
                            self.dragging, self.last_mouse_pos = True, mouse
                if event.type == pygame.MOUSEBUTTONUP: self.dragging = False
                if event.type == pygame.MOUSEMOTION and self.dragging:
                    dx, dy = mouse[0] - self.last_mouse_pos[0], mouse[1] - self.last_mouse_pos[1]
                    self.offset_x, self.offset_y, self.last_mouse_pos = self.offset_x + dx, self.offset_y + dy, mouse
                if event.type == pygame.MOUSEWHEEL and self.state == "MAZE":
                    self.zoom = max(0.5, min(self.zoom + event.y * 0.1, 10.0))
                if event.type == pygame.KEYDOWN and self.state == "MENU" and self.active_input:
                    if event.key == pygame.K_BACKSPACE:
                        if self.active_input == "X": self.input_x = self.input_x[:-1]
                        else: self.input_y = self.input_y[:-1]
                    elif event.unicode.isdigit():
                        if self.active_input == "X": self.input_x += event.unicode
                        else: self.input_y += event.unicode

            if self.state == "MENU":
                self.draw_menu(mouse)
            else:
                if self.generating:
                    for _ in range(max(1, (self.grid_x*self.grid_y)//150)): self.prim_step()
                if self.solving:
                    if self.compute_mode == "CPU_ASTAR":
                        for _ in range(max(1, (self.grid_x*self.grid_y)//200)):
                            if not self._timed_astar_step(): break
                    elif self.compute_mode == "GPU_FLOOD" and self._flood_state:
                        self._step_flood_gpu()
                self.draw_maze_screen(mouse)
            pygame.display.flip()
            clock.tick(60)

    def handle_clicks(self, pos):
        if self.state == "MENU":
            if pygame.Rect(400, 150, 100, 40).collidepoint(pos): self.active_input = "X"
            elif pygame.Rect(600, 150, 100, 40).collidepoint(pos): self.active_input = "Y"
            elif pygame.Rect(630, 265, 22, 22).collidepoint(pos): self.visual_gen = not self.visual_gen
            elif pygame.Rect(450, 250, 160, 50).collidepoint(pos): self.init_maze(int(self.input_x or 20), int(self.input_y or 20))
            else:
                self.active_input = None
                presets = [(5,5), (20,5), (30,30), (70,50)]
                for i, p in enumerate(presets):
                    if pygame.Rect(320 + (i%2)*250, 350 + (i//2)*250, 210, 210).collidepoint(pos): self.init_maze(p[0], p[1])
        else:
            if pygame.Rect(20, 25, 120, 45).collidepoint(pos): self.state = "MENU"
            if pygame.Rect(SCREEN_WIDTH-130, 25, 110, 45).collidepoint(pos): self.init_maze(self.grid_x, self.grid_y)
            if pygame.Rect(SCREEN_WIDTH-280, 35, 22, 22).collidepoint(pos):
                self.visual_gen = not self.visual_gen
                self.init_maze(self.grid_x, self.grid_y)
            if pygame.Rect(SCREEN_WIDTH-480, SCREEN_HEIGHT-65, 22, 22).collidepoint(pos): self.path_visual = not self.path_visual
            if pygame.Rect(SCREEN_WIDTH-300, SCREEN_HEIGHT-75, 160, 45).collidepoint(pos):
                self.compute_mode = "GPU_FLOOD" if self.compute_mode == "CPU_ASTAR" and self.gpu_available else "CPU_ASTAR"
            if pygame.Rect(20, SCREEN_HEIGHT-75, 130, 45).collidepoint(pos): self.randomize_points()
            if pygame.Rect(SCREEN_WIDTH-130, SCREEN_HEIGHT-75, 110, 45).collidepoint(pos):
                self.solve_compute_time, self.last_solve_time, self.last_solve_method, self._flood_state, self.solving, self.path, self.searched = 0.0, None, None, None, False, [], set()
                self.start_astar()

    def draw_menu(self, mouse):
        txt = FONT_L.render("MAZE CONFIGURATOR", True, NV_GREEN)
        SCREEN.blit(txt, (SCREEN_WIDTH//2 - txt.get_width()//2, 70))
        for i, (label, val, rect) in enumerate([("X", self.input_x, pygame.Rect(400, 150, 100, 40)), ("Y", self.input_y, pygame.Rect(600, 150, 100, 40))]):
            active = (self.active_input == label)
            color = NV_GREEN if active else WHITE
            pygame.draw.rect(SCREEN, (30,30,30), rect)
            pygame.draw.rect(SCREEN, color, rect, 2)
            SCREEN.blit(FONT_M.render(val + ("|" if active and self.cursor_timer < 30 else ""), True, color), (rect.x+10, rect.y+8))
        self.draw_checkbox(630, 265, "Visualization", self.visual_gen, mouse)
        self.draw_nv_btn(pygame.Rect(450, 250, 160, 50), "GENERATE", NV_GREEN)
        presets = [("5x5", 5, 5), ("20x5", 20, 5), ("30x30", 30, 30), ("70x50", 70, 50)]
        for i, (lab, px, py) in enumerate(presets):
            box = pygame.Rect(320 + (i%2)*250, 350 + (i//2)*250, 210, 210)
            color = NV_GREEN if box.collidepoint(mouse) else WHITE
            pygame.draw.rect(SCREEN, (20,20,20), box)
            pygame.draw.rect(SCREEN, color, box, 2)
            SCREEN.blit(FONT_M.render(lab, True, color), (box.centerx - 20, box.y + 10))
            self.draw_mini_maze(pygame.Rect(box.x+20, box.y+40, 170, 150), self.previews[(px, py)], px, py)

    def draw_maze_screen(self, mouse):
        ui_overlay = pygame.Surface((SCREEN_WIDTH, UI_TOP_HEIGHT), pygame.SRCALPHA)
        ui_overlay.fill((*NV_GREEN, 51))
        SCREEN.blit(ui_overlay, (0, 0))
        SCREEN.blit(ui_overlay, (0, SCREEN_HEIGHT - UI_BOT_HEIGHT))

        top_text = f"{self.last_solve_method} — {self.last_solve_time*1000.0:.2f} ms" if self.last_solve_time else (f"{self.last_solve_method} — running..." if self.last_solve_method else "No solve yet")
        txt = FONT_M.render(top_text, True, WHITE)
        SCREEN.blit(txt, (SCREEN_WIDTH//2 - txt.get_width()//2, 12))

        self.draw_nv_btn(pygame.Rect(20, 25, 120, 45), "  MENU", NV_GREEN, "back")
        self.draw_nv_btn(pygame.Rect(SCREEN_WIDTH-130, 25, 110, 45), "REGEN", NV_GREEN)
        self.draw_checkbox(SCREEN_WIDTH-280, 35, "Visual", self.visual_gen, mouse)
        self.draw_checkbox(SCREEN_WIDTH-480, SCREEN_HEIGHT-65, "Path Visual", self.path_visual, mouse)
        self.draw_nv_btn(pygame.Rect(SCREEN_WIDTH-300, SCREEN_HEIGHT-75, 160, 45), ("GPU : Flood" if self.compute_mode=="GPU_FLOOD" else "CPU : A*"), NV_GREEN)
        self.draw_nv_btn(pygame.Rect(20, SCREEN_HEIGHT-75, 130, 45), "RANDOMIZE", NV_GREEN)
        self.draw_nv_btn(pygame.Rect(SCREEN_WIDTH-130, SCREEN_HEIGHT-75, 110, 45), "SOLVE", NV_GREEN)

        m_top, m_bot = UI_TOP_HEIGHT + 20, SCREEN_HEIGHT - UI_BOT_HEIGHT - 20
        c_size = min((SCREEN_WIDTH - 40) / self.grid_x, (m_bot - m_top) / self.grid_y) * self.zoom
        ox, oy = (SCREEN_WIDTH - (c_size * self.grid_x)) // 2 + self.offset_x, m_top + ((m_bot - m_top) - (c_size * self.grid_y)) // 2 + self.offset_y

        SCREEN.set_clip(pygame.Rect(0, UI_TOP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT - UI_TOP_HEIGHT - UI_BOT_HEIGHT))
        
        # --- LAYER 1: Searched Area ---
        road_surf = pygame.Surface((math.ceil(c_size), math.ceil(c_size)), pygame.SRCALPHA)
        road_surf.fill((*DARK_GREEN, 255))
        for x, y in self.searched: 
            SCREEN.blit(road_surf, (ox + x*c_size, oy + y*c_size))

        # --- LAYER 2: Final Path ---
        path_color = RED if max(self.grid_x, self.grid_y) > 100 else NV_GREEN
        path_surf = pygame.Surface((math.ceil(c_size), math.ceil(c_size)), pygame.SRCALPHA)
        path_surf.fill((*path_color, 255))
        if self.path:
            for x, y in self.path: 
                SCREEN.blit(path_surf, (ox + x*c_size, oy + y*c_size))

        # --- LAYER 3: Start/Goal Indicators ---
        pygame.draw.rect(SCREEN, YELLOW, (ox + self.start_pos[0]*c_size, oy + self.start_pos[1]*c_size, math.ceil(c_size), math.ceil(c_size)))
        pygame.draw.rect(SCREEN, ORRANGE, (ox + self.goal_pos[0]*c_size, oy + self.goal_pos[1]*c_size, math.ceil(c_size), math.ceil(c_size)))

        # --- LAYER 4: Walls  ---
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                px, py, w = ox + x*c_size, oy + y*c_size, self.cells[x][y].walls
                # Drawing lines with a width of 2 for better visibility
                if w[0]: pygame.draw.line(SCREEN, GREEN, (px, py), (px+c_size, py), 2)
                if w[1]: pygame.draw.line(SCREEN, GREEN, (px+c_size, py), (px+c_size, py+c_size), 2)
                if w[2]: pygame.draw.line(SCREEN, GREEN, (px, py+c_size), (px+c_size, py+c_size), 2)
                if w[3]: pygame.draw.line(SCREEN, GREEN, (px, py), (px, py+c_size), 2)

        SCREEN.set_clip(None)

if __name__ == "__main__":
    app = MazeApp()
    app.run()