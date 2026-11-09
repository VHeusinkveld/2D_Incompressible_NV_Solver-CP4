# Numerical corrections and validation

The solver retains its staggered MAC grid, explicit conservative advection with
central/donor-cell blending, backward-Euler viscosity, and pressure projection.
Time integration is first order. Solid geometry is represented by pressure cells
on a Cartesian staircase grid.

## Pressure projection and obstacles

Let `D` be the discrete divergence, `G` the pressure gradient, and `M` the mask
that keeps active velocity faces. The pressure matrix is now assembled as
`A = -D M G`. We solve `A p = -D u_star / dt`, then apply
`u_new = M (u_star - dt G p)`. This makes divergence cancellation hold for exactly
the faces on which velocity is actually updated.

The previous obstacle routine removed pressure couplings of size `1/h²` but
adjusted the diagonal by `dt/(Re h²)`. It also enlarged the velocity masks beyond
the pressure geometry. Correcting only the diagonal therefore did not fix the
projection. The new implementation blocks a face when either adjacent pressure
cell is solid and constructs the pressure matrix from those same faces.

There is one pressure reference per connected component, including isolated
solid cells. The right-hand side is checked for flux compatibility in every
component before fixing its reference value. Inconsistent prescribed inflow and
outflow raise an error instead of hiding mass imbalance in a pressure gauge.

Normal velocities on solid faces are zero. For tangential diffusion next to a
half-cell wall, the ghost velocity is reflected (`u_ghost = -u_fluid`), imposing
stationary no-slip at the wall rather than one extra half-cell inside the solid.
Geometry touching an outer boundary or unresolved on the grid is rejected.

## Boundary conditions and initialization

All existing named modes prescribe fixed velocities. `horizontal_sea` prescribes
`u=velocity, v=0` on all outer boundaries. `horizontal_tube` additionally makes
the north/south walls stationary. `lid` keeps the normal wall velocity zero and
moves the north lid. The legacy `vertical_tube` mode prescribes uniform vertical
velocity at all outer boundaries; despite its name it is vertical free-stream
flow. `airfoil` is the legacy all-zero boundary mode, not an inflow preset.

The unconditional opposite-edge velocity copying was removed. It was neither a
consistent periodic discretization nor a fixed inflow/outflow condition. Periodic
and radiation/open-outlet modes are not implemented. Unknown boundary modes
raise an error. The retained `update_BC` helper does not change fixed boundaries.

North/south diffusion forcing uses `hy²`; east/west uses `hx²`. The old swap was
invisible for square cells. Each angle/velocity case in the notebook now creates
a new velocity state. NACA outlines use array concatenation and polygon filling
at cell centres, fixing symmetric-profile errors and rounded-outline gaps.

## Visualization

`plot_Q` now plots streamlines directly from the computed velocity field. The
previous auxiliary streamfunction code used pressure-grid indices on a corner
grid, read diagonal entries from the wrong matrix, scaled by `1/dt`, and imposed
zero outer streamfunction even for through-flow. The new video shows signed
vorticity `dv/dx - du/dy`, with fixed colour limits across all frames.

## Automated checks

Run `python -m unittest discover -s tests -v`. Fourteen tests cover:

- Multi-step incompressibility with no object, a circle, and a rotated NACA
  profile, using both cached factorizations and direct solves.
- Equality of the pressure matrix and independently applied divergence/gradient
  operators, independence from viscosity/timestep, removal of a manufactured
  pure-gradient velocity, and gauges for disconnected regions.
- Rejection of incompatible flux and an unsafe explicit-advection CFL number.
- Preservation of fixed boundaries through the complete simulation wrapper.
- Uniform horizontal/vertical flow on unequal cell widths.
- An analytic discrete diffusion eigenmode and a linear profile at an internal
  half-cell no-slip wall.
- Symmetric NACA outlines, rotated geometry, and preservation of cylinder-flow
  reflection symmetry without an imposed perturbation.

The tests exercise SciPy's direct and cached LU paths in environments without
CHOLMOD. When CHOLMOD is installed, the cached path selects it. An automated
GitHub Actions workflow runs the same test suite on Python 3.12.

## New cylinder video

`python render_wake.py` runs the same `simulation_step` function as the notebook.
It uses `U=1`, `D=1`, `nu=1/150`, a `20D × 8D` domain, a cylinder centred at
`(4D,4D)`, `400 × 160` cells, `dt=0.005D/U`, and final time `60D/U`.
The circle spans 20 cells in diameter. All four outer boundaries prescribe
`u=1, v=0`. The frame crops to `x=0..18D`, `y=1..7D`; the computation uses the
entire domain.

A single compact streamfunction perturbation of amplitude 0.02 is added in
`x=5..7D, y=3..5D` at initialization. Its discrete curl is divergence-free and
breaks the exact reflection symmetry so the wake can develop. There is no
ongoing artificial vorticity forcing. The first velocity step includes the
projection of the initial free stream around the obstacle.

Every step checks divergence, CFL, net boundary flux, and blocked-face velocity.
The run aborts if maximum divergence exceeds `1e-7`; the solver rejects CFL above
1. The JSON records maxima across all steps, not only saved frames. The CSV
contains every displayed frame's diagnostics and the cross-stream velocity at
the probe near `(6.5D,4.4D)`. The MP4 uses H.264, 1280 × 720 pixels and 24 fps.

These checks establish discrete consistency and stability for this run. They
are not spatial, temporal, or domain-convergence studies. Numerical diffusion,
the staircase surface, 12.5% blockage, and the distant fixed outlet affect the
solution. In particular, late vortices can interact with the outlet. Do not use
this video or the legacy pressure-force estimator as validated drag, lift, or
Strouhal data. The historical movies and saved result arrays were not recomputed.
