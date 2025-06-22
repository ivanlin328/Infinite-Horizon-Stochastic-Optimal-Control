# Trajectory‑Tracking with CEC & GPI  
_ECE 276B Project – Receding‑Horizon Chance‑Constrained Control vs. Grid Policy Iteration_

## 1 . Overview  
This repository contains a full simulation and benchmarking stack for a non‑holonomic mobile robot (a simple car) tasked with following a 2‑D reference trajectory while avoiding circular obstacles.

Two distinct controllers are compared:

| Controller | File | Idea in one line |
|------------|------|------------------|
| **CEC** (Chance‑constrained Explicit Controller) | [`cec.py`](cec.py) | Finite‑horizon nonlinear program solved every Δ _s_ with chance constraints on collision avoidance |
| **GPI** (Grid Policy Iteration) | [`gpi.py`](gpi.py) | Tabular value‑iteration / policy‑iteration on a discretised error state grid, accelerated with **Ray** |

`main.py` orchestrates the simulation, logging and visualisation.

<p align="center"><img src="assets/demo.gif" width="480"/></p>

---

## 2 . Repository structure
```
├── main.py               # Entry point – toggle `use_cec` / `use_gpi`
├── cec.py                # Receding‑horizon CEC controller
├── gpi.py                # Grid‑based PI controller (+ dataclass config)
├── value_function.py     # Tabular value‑function helpers
├── utils.py              # Trajectory generator, visualisation, helpers
├── mujoco_car.py         # (optional) MuJoCo visualisation wrapper
└── README.md             # <–– you are here
```
> **Note:** `mujoco_car.py` is optional – if absent the simulation still
> runs headless and produces Matplotlib plots/GIFs.

---

## 3 . Prerequisites
Create a virtual environment (conda or `python -m venv`) and install:

```bash
pip install -r requirements.txt
```

<details><summary>requirements.txt</summary>

```
numpy
matplotlib
scipy
tqdm
casadi
ray[default]
mujoco==3.2.*            # optional – only if you use MuJoCo visualiser
mujoco-python            # optional
```
</details>

Python ≥ 3.9 is recommended (tested with 3.10).

---

## 4 . Quick start
```bash
# Run the CEC controller (fast, 3–4 FPS on laptop CPU)
python main.py            # default: use_cec=True, use_gpi=False

# Run the GPI controller (slow – pre‑computes value‑function)
# open main.py and set:   use_gpi = True
python main.py

# Both controllers share the same log/plot pipeline:
# ‑ Plots saved to   outputs/cec_*.png   or   outputs/gpi_*.png
# ‑ A GIF (trajectory animation) is written next to the plots
```

After the run finishes you should see console timing statistics and a GIF
similar to the demo above.

---

## 5 . Parameter tuning

All high‑level hyper‑parameters live at the top of **`main.py`**:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `delta`   | Sampling interval Δ _s_ | `utils.time_step` (0.5 s) |
| `horizon` | CEC planning horizon **T** | `15` |
| `γ`       | Discount factor (GPI) | `0.98` |
| `Q, q, R` | Stage‑cost weights | see code |
| `obstacles` | `np.ndarray([[x, y, r], …])` | example in `main.py` |
| `r_robot` | Robot radius (m) | `0.3` |

To change the reference path adapt `utils.lissajous` or plug‑in your own
trajectory generator.

---

## 6 . Extending
* Swap the simple P‑controller in `utils.simple_controller` for your own.
* Increase grid resolution in `gpi.py` (`ex_space`, `ey_space`, …).
* Enable MuJoCo visualisation by setting `use_mujoco = True` and ensuring
  `mujoco_car.py` plus MuJoCo 3.2 libraries are installed.

---

## 7 . Troubleshooting
| Symptom | Fix |
|---------|-----|
| **`ModuleNotFoundError: mujoco`** | Set `use_mujoco = False` or install MuJoCo |
| **Ray “address already in use”** | `ray stop --force` before rerunning |
| **Long GPI pre‑compute time** | Reduce grid points or `num_evals` |

---

## 8 . License
MIT – see `LICENSE` (add your name/year).

---

## 9 . Acknowledgements
Course **ECE 276B – Planning & Learning in Robotics** at UC San Diego.

> 🍀 Happy trajectory tracking!
