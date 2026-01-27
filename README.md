# 🟩 Maze Lab — Interactive Maze Generator & Solver

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2-orange)](https://www.pygame.org/news)
[![Numba](https://img.shields.io/badge/Numba-CUDA-green)](https://numba.pydata.org/)

---

## 🚀 Project Overview

**Maze Lab** is an interactive Python application for generating and solving mazes. It combines **procedural maze generation** with **pathfinding algorithms**, including CPU-based A* and GPU-accelerated flood-fill, to visualize and explore algorithms in real-time.

![Maze Example](Images/MazeLab.png)

---

## 🌟 Key Features

- **Dynamic Maze Generation:** Procedural mazes using **Prim’s algorithm**, supporting custom grid sizes.
- **Dual Engine Pathfinding:**
  - **CPU A\***: Optimized heuristic search for precision pathfinding.
  - **GPU Flood-Fill**: Parallel wavefront expansion using **Numba + CUDA** (if GPU available).
- **Real-Time Visualization:** Toggle maze generation, pathfinding visualization, and see the algorithms in action.
- **Interactive GUI:** Zoom, pan, and adaptive rendering of the maze.
- **Performance Metrics:** Real-time **timing display** to compare algorithm efficiency.


<div style="display: flex; justify-content: center;">
  <img src="Images/Prim.png" width="300" style="margin-right: 10px;">
  <img src="Images/Flood.png" width="300">
</div>



---

## 🎮 Controls

* **Mouse Wheel:** Zoom In / Out.
* **Left Click + Drag:** Pan across the maze.
* **REGEN:** Create a new maze architecture.
* **SOLVE:** Execute the selected pathfinding engine.
* **GPU/CPU Toggle:** Switch between hardware acceleration modes.
* **RANDOMIZE:** Relocate the Start and Goal points instantly.

---

## 🛠️ Requirements

- **Python 3.x**  
- **Pygame**  
- **Numba** (for GPU acceleration, optional)  
- **CUDA Toolkit** (if using GPU acceleration)

Install dependencies:

```bash
pip install pygame numba
