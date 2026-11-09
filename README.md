# Navier-Stokes: NACA-profile objects in 2D flow

## NACA wing video

[![NACA 2412 wing](movies/naca_wake_corrected.png)](movies/naca_wake_corrected.mp4)

The requested wing is **NACA 2412 at 30°**, matching the notebook profile and first angle. Chord 2, Reynolds number 300 based on chord, 400 × 160 cells, and no imposed perturbation. Reproduce with `OPENBLAS_NUM_THREADS=1 python render_naca.py`. Settings and diagnostics accompany the MP4. This remains a finite-grid demonstration, not a converged aerodynamic benchmark.

## Cylinder validation video

[![Corrected cylinder wake](movies/cylinder_wake_corrected.png)](movies/cylinder_wake_corrected.mp4)

[Watch the corrected cylinder wake](movies/cylinder_wake_corrected.mp4). The video
shows signed vorticity and a downstream velocity probe, with divergence and CFL
reported on every frame. It uses a 400 × 160 grid, a unit-diameter cylinder,
Reynolds number 150 based on that diameter, and 60 flow-time units.

The obstacle pressure correction now uses the same blocked faces as the velocity
update. Fixed boundaries remain fixed, pressure has an explicit reference, and
parameter sweeps start each case independently. See [numerical details and
validation](NUMERICS.md), [run metadata](movies/cylinder_wake_corrected.json), and
[frame diagnostics](movies/cylinder_wake_corrected.csv).

This is a numerical wake demonstration. The staircase geometry, finite domain,
and fixed far-field/outlet velocities have not been validated as a converged
force or shedding-frequency benchmark.

## Run

Python 3.10+:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python render_wake.py
```

Install FFmpeg and put it on `PATH` to encode the MP4. The simulation can run
without it using `python render_wake.py --simulate-only`; use `--render-only` to
render the cached fields afterwards. Intermediate fields are kept in `.cache/`
and are not committed. The script saves the MP4, a final-frame PNG, a diagnostics
CSV, and JSON containing settings, dependency versions, and source hashes.

`Execute_simulation.ipynb` remains the entry point for NACA angle/velocity sweeps.
Its saved 1200 × 600 setup is expensive; lower `multi_arr` for exploratory runs.
`const.cholesky=True` caches sparse factorizations, using CHOLMOD when installed
and SciPy LU otherwise. `False` uses direct SciPy solves on each step.

## Historical video

The earlier video below predates the numerical corrections and is retained as
historical output.

[![Simulation result](https://img.youtube.com/vi/TGbapm781wk/0.jpg)](https://www.youtube.com/watch?v=TGbapm781wk)

## Structure

*create_animation.ipynb*: Contains function for making a animation from premade figures

*Execute_simulation.ipynb*: Main file which executes the NS simulation. This contains the set-up of the simulation.

---

*constants.py*: Contains the constants of the system

*data_processing.py*: Contains plotting functions

*functions.py*: Contains all functions needed to perform the simulation (generated the wanted data).

*simulation.py*: Contains the simulation functions of the NS solver

*operators.py*: Compatible pressure operator, obstacle diffusion, and diagnostics.

*render_wake.py*: Reproducible cylinder simulation and video rendering.

*tests/test_solver.py*: Numerical regression checks, also run in GitHub Actions.
