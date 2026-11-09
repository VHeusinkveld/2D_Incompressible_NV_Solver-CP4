"""Reproduce the cylinder-wake video with the same solver used by the notebook.

Run: OPENBLAS_NUM_THREADS=1 python render_wake.py
FFmpeg is required for rendering, but not for --simulate-only.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import scipy

from constants import constants
from functions import (initialisation, set_BC, set_BM, creat_obj, laplacian,
                       laplace_obj, cholesky_decomposition)
from operators import cell_velocity, divergence
from simulation import simulation_step

ROOT = Path(__file__).resolve().parent
SOLVER_FILES = ('constants.py', 'functions.py', 'operators.py', 'simulation.py', 'render_wake.py')


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOLVER_FILES}


def simulate(args):
    settings = dict(nx=args.nx, ny=args.ny, lx=20., ly=8., dt=args.dt,
                    tf=args.tf, Re=args.re, velocity=1., rho_arrow=10, cholesky=True)
    c = constants(NS(**settings))
    settings.update(dt=c.dt, nt=c.nt)
    o = creat_obj(c, NS(sort='circle', radius=.5, center=(4., 4.)))
    bc = set_BM(c, set_BC(c, NS(), 'horizontal_sea'))
    lp = cholesky_decomposition(laplace_obj(c, laplacian(c, NS()), o))
    d = initialisation(c, NS())
    d.U[:] = ~o.Ugrid
    # A single small, compact divergence-free perturbation seeds the otherwise
    # exactly symmetric wake. It is never applied again during time integration.
    x, y = c.x[:, None], c.y[None, :]
    sx = np.sin(np.pi * np.clip((x - 5.) / 2., 0, 1))**2
    sy = np.sin(np.pi * np.clip((y - 3.) / 2., 0, 1))**2
    psi = .02 * sx * sy
    d.U += np.diff(psi, axis=1)[1:-1, :] / c.hy
    d.V -= np.diff(psi, axis=0)[:, 1:-1] / c.hx
    d.U[o.Ugrid] = 0
    d.V[o.Vgrid] = 0
    probe = (int(6.5/c.hx), int(4.4/c.hy))
    stride = max(1, round(.1 / c.dt))
    fields, records = [], []
    max_div, max_cfl, max_flux, max_wall = 0., 0., 0., 0.
    started = time.perf_counter()
    for step in range(1, c.nt + 1):
        simulation_step(c, bc, o, lp, d)
        div = divergence(c, bc, d)
        net_flux = float(div.sum() * c.hx * c.hy)
        wall = max(float(np.max(np.abs(d.U[o.Ugrid]))),
                   float(np.max(np.abs(d.V[o.Vgrid]))))
        max_div = max(max_div, d.max_divergence)
        max_cfl = max(max_cfl, d.cfl)
        max_flux = max(max_flux, abs(net_flux))
        max_wall = max(max_wall, wall)
        if d.max_divergence > 1e-7:
            raise RuntimeError(f'Projection failed at step {step}: divergence={d.max_divergence}')
        if step == 1 or step % stride == 0 or step == c.nt:
            # Signed vorticity on the MAC corner grid; physical units U/D.
            omega = np.diff(d.V, axis=0)/c.hx - np.diff(d.U, axis=1)/c.hy
            omega[o.Qgrid] = np.nan
            fields.append(omega.astype(np.float32))
            _, vc = cell_velocity(c, bc, d)
            records.append((step*c.dt, d.max_divergence, d.cfl, net_flux, wall, float(vc[probe])))
        if step % max(1, c.nt // 20) == 0:
            print(f'{step/c.nt:5.0%}  t={step*c.dt:6.2f}  '
                  f'max|div|={d.max_divergence:.2e}  CFL={d.cfl:.3f}  '
                  f'elapsed={time.perf_counter()-started:.0f}s', flush=True)
    metadata = {
        'case': 'Stationary cylinder, 2D laminar wake demonstration',
        'settings': settings, 'diameter': 1., 'center': [4., 4.],
        'reynolds_based_on_diameter': c.velocity * 1. * c.Re,
        'boundaries': 'Fixed u=1, v=0 at all four outer boundaries; not periodic or a radiation outlet.',
        'initial_condition': 'Uniform u=1 outside the cylinder, v=0; compact streamfunction seed of amplitude 0.02 at x=5..7, y=3..5; first step projects velocity.',
        'geometry': 'Staircase circle on the pressure grid; normal wall velocity zero, tangential ghost velocity reflected.',
        'interpretation': 'Numerical demonstration, not a grid/domain-converged Strouhal or force benchmark. The distant fixed outlet can affect the late wake.',
        'probe_xy': [6.5, 4.4], 'frames': len(fields),
        'max_abs_divergence_all_steps': max_div, 'max_cfl_all_steps': max_cfl,
        'max_abs_net_boundary_flux_all_steps': max_flux,
        'max_abs_blocked_face_velocity_all_steps': max_wall,
        'wall_seconds': time.perf_counter()-started,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
        'source_sha256': source_hashes(),
    }
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, omega=np.asarray(fields), records=np.asarray(records),
                        solid=o.Pgrid, metadata=json.dumps(metadata))
    return cache


def render(cache, output, fps):
    with np.load(cache) as saved:
        fields, records, solid = saved['omega'], saved['records'], saved['solid']
        meta = json.loads(str(saved['metadata']))
    if meta['source_sha256'] != source_hashes():
        raise RuntimeError('Cached fields came from different source files; rerun the simulation.')
    if not FFMpegWriter.isAvailable():
        raise RuntimeError('Install FFmpeg to render the video; simulated fields are saved in the cache.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    c = NS(**meta['settings'])
    hx, hy = c.lx/c.nx, c.ly/c.ny
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': '#eaf0f9', 'axes.labelcolor': '#aebed2',
                         'xtick.color': '#aebed2', 'ytick.color': '#aebed2',
                         'axes.edgecolor': '#3c4e66', 'savefig.facecolor': '#101a29'})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='#101a29')
    fig.text(.065, .936, 'A cylinder wake, recomputed', fontsize=24, weight='bold')
    fig.text(.065, .893, f'2D Navier–Stokes  |  Re$_D$ = {meta["reynolds_based_on_diameter"]:g}  |  '
             f'{c.nx} × {c.ny} cells  |  Δt = {c.dt:g} D/U', color='#aebed2', fontsize=12)
    ax = fig.add_axes([.065, .37, .80, .46], facecolor='#f4f6f8')
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#233348')
    im = ax.imshow(fields[0].T, origin='lower', interpolation='nearest',
                   extent=(hx/2, c.lx-hx/2, hy/2, c.ly-hy/2),
                   cmap=cmap, vmin=-5, vmax=5, aspect='equal')
    ax.set_xlim(0, 18)
    ax.set_ylim(1, 7)
    ax.set_xlabel('x / D')
    ax.set_ylabel('y / D')
    ax.plot(6.5, 4.4, 'o', ms=4, mfc='none', mec='#28384b', mew=1)
    ax.annotate('Flow', xy=(2.1, 6.3), xytext=(.7, 6.3), color='#28384b',
                va='center', arrowprops={'arrowstyle': '->', 'color': '#28384b'})
    cax = fig.add_axes([.89, .41, .014, .34])
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label('Signed vorticity  ωD / U', labelpad=12)
    trace = fig.add_axes([.065, .115, .80, .15], facecolor='#142238')
    trace.set_xlim(0, c.tf)
    probe_limit = max(.1, 1.15*np.max(np.abs(records[:, 5])))
    trace.set_ylim(-probe_limit, probe_limit)
    trace.axhline(0, color='#455670', lw=.7)
    trace.grid(axis='x', color='#2d3e56', lw=.5)
    trace.set_ylabel('Probe v / U')
    trace.set_xlabel('Simulation time  tU / D')
    line, = trace.plot([], [], color='#65d3e4', lw=1.5)
    trace.set_title('Cross-stream velocity behind the cylinder', loc='left',
                    color='#aebed2', fontsize=10, pad=8)
    time_label = fig.text(.065, .838, '', fontsize=11, family='monospace')
    fig.text(.065, .025, 'One initial perturbation seeds the wake. Fixed far-field velocities; '
             'this is a numerical demonstration, not a converged benchmark.',
             fontsize=9, color='#aebed2')
    writer = FFMpegWriter(fps=fps, codec='libx264',
                         extra_args=['-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    started = time.perf_counter()
    with writer.saving(fig, str(output), dpi=100):
        for i, (field, row) in enumerate(zip(fields, records)):
            im.set_data(field.T)
            line.set_data(records[:i+1, 0], records[:i+1, 5])
            time_label.set_text(f't = {row[0]:5.2f} D/U     max |div u| = {row[1]:.1e}     CFL = {row[2]:.3f}')
            writer.grab_frame()
            if i == len(fields)-1:
                fig.savefig(output.with_suffix('.png'), dpi=100)
    plt.close(fig)
    meta.update(video_fps=fps, video_duration_seconds=len(fields)/fps,
                video_codec='H.264 / yuv420p', rendering_seconds=time.perf_counter()-started)
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time', 'max_abs_divergence', 'cfl', 'net_boundary_flux',
                         'max_blocked_face_velocity', 'probe_v'])
        writer.writerows(records)
    print(f'Rendered {output} ({output.stat().st_size/1e6:.2f} MB)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=400)
    parser.add_argument('--ny', type=int, default=160)
    parser.add_argument('--dt', type=float, default=.005)
    parser.add_argument('--tf', type=float, default=60.)
    parser.add_argument('--re', type=float, default=150.)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--cache', default=str(ROOT/'.cache/cylinder_wake.npz'))
    parser.add_argument('--output', default=str(ROOT/'movies/cylinder_wake_corrected.mp4'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--render-only', action='store_true')
    mode.add_argument('--simulate-only', action='store_true')
    args = parser.parse_args()
    cache = Path(args.cache) if args.render_only else simulate(args)
    if not args.simulate_only:
        render(cache, args.output, args.fps)


if __name__ == '__main__':
    main()
