"""Render a corrected equivalent of the historical pressure/velocity wing movie.

OPENBLAS_NUM_THREADS=1 python render_naca_pressure.py
The historical video's exact run parameters are not recoverable from the MP4.
This explicitly specified case matches its 2 x 1 framing and visualization.
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
SOURCES = ('constants.py', 'functions.py', 'operators.py', 'simulation.py',
           'render_naca_pressure.py')


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def relative_pressure(pressure, solid):
    """Remove the arbitrary gauge using the fluid mean in the inlet 10% strip."""
    strip = max(1, pressure.shape[0]//10)
    reference = float(np.mean(pressure[:strip][~solid[:strip]]))
    result = pressure.copy() - reference
    result[solid] = np.nan
    return result, reference


def simulate(args):
    settings = dict(nx=args.nx, ny=args.ny, lx=2., ly=1., dt=args.dt,
                    tf=args.tf, Re=args.re, velocity=1., rho_arrow=20, cholesky=True)
    c = constants(NS(**settings))
    settings.update(dt=c.dt, nt=c.nt)
    o = creat_obj(c, NS(sort='naca', profile='2412', angle=30., chord=.5, center=(1., .5)))
    bc = set_BM(c, set_BC(c, NS(), 'horizontal_sea'))
    lp = cholesky_decomposition(laplace_obj(c, laplacian(c, NS()), o))
    d = initialisation(c, NS())
    d.U[:] = ~o.Ugrid
    # Uniform initial flow, with no imposed perturbation; the first step projects it.
    ix = np.unique(np.linspace(1, c.nx-2, min(39, c.nx-2)).astype(int))
    iy = np.unique(np.linspace(1, c.ny-2, min(19, c.ny-2)).astype(int))
    arrow_index = np.ix_(ix, iy)
    arrow_solid = o.Pgrid[arrow_index]
    stride = max(1, round(args.frame_interval / c.dt))
    pressures, us, vs, records = [], [], [], []
    max_div = max_cfl = max_flux = max_wall = 0.
    started = time.perf_counter()
    for step in range(1, c.nt+1):
        simulation_step(c, bc, o, lp, d)
        flux = float(divergence(c, bc, d).sum()*c.hx*c.hy)
        wall = max(float(np.max(np.abs(d.U[o.Ugrid]))), float(np.max(np.abs(d.V[o.Vgrid]))))
        max_div = max(max_div, d.max_divergence)
        max_cfl = max(max_cfl, d.cfl)
        max_flux = max(max_flux, abs(flux))
        max_wall = max(max_wall, wall)
        if d.max_divergence > 1e-7 or not np.isfinite(d.P).all():
            raise RuntimeError(f'Invalid field at step {step}: div={d.max_divergence}')
        if step % stride == 0 or step == c.nt:
            p, reference = relative_pressure(d.P, o.Pgrid)
            u, v = cell_velocity(c, bc, d)
            u, v = u[arrow_index].copy(), v[arrow_index].copy()
            u[arrow_solid], v[arrow_solid] = np.nan, np.nan
            pressures.append(p.astype(np.float32))
            us.append(u.astype(np.float32))
            vs.append(v.astype(np.float32))
            records.append((step*c.dt, d.max_divergence, d.cfl, flux, wall, reference,
                            float(np.nanmin(p)), float(np.nanmax(p))))
        if step % max(1, c.nt//20) == 0:
            print(f'{step/c.nt:5.0%} t={step*c.dt:5.2f} div={d.max_divergence:.2e} '
                  f'CFL={d.cfl:.3f} elapsed={time.perf_counter()-started:.0f}s', flush=True)
    metadata = {
        'case': 'NACA 2412 pressure and velocity, historical-video framing',
        'reference_video': 'https://www.youtube.com/watch?v=TGbapm781wk',
        'reference_repo_video': 'movies/Tube_flow_nana_10e8_600_300.mp4',
        'comparison': 'Same visualization and 2 x 1 domain framing; newly specified run, not an exact parameter replay. Historical Reynolds number and timestep are unverified.',
        'settings': settings, 'profile': '2412', 'angle_degrees_clockwise': 30.,
        'chord': .5, 'center': [1., .5], 'cells_per_chord': .5/c.hx,
        'reynolds_based_on_chord': .5*c.velocity*c.Re,
        'boundaries': 'Fixed u=1, v=0 on all outer boundaries; no periodic or open outlet.',
        'initial_condition': 'Uniform flow outside the wing; no imposed perturbation; first step projects velocity.',
        'pressure': 'p/rho relative to the instantaneous fluid mean in the inlet 10% of the domain; masked inside the wing.',
        'velocity_arrows': 'Cell-centred velocity; fixed scale for all frames; masked inside the wing.',
        'interpretation': 'Visualization demonstration, not a converged aerodynamic benchmark. Compact fixed boundaries influence the solution.',
        'frames': len(records), 'first_frame_time': records[0][0],
        'sample_interval': stride*c.dt,
        'max_abs_divergence_all_steps': max_div, 'max_cfl_all_steps': max_cfl,
        'max_abs_net_boundary_flux_all_steps': max_flux,
        'max_abs_blocked_face_velocity_all_steps': max_wall,
        'simulation_seconds': time.perf_counter()-started,
        'source_sha256': source_hashes(),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
    }
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pressure=np.asarray(pressures), u=np.asarray(us), v=np.asarray(vs),
                        arrow_x=(ix+.5)*c.hx, arrow_y=(iy+.5)*c.hy,
                        records=np.asarray(records), solid=o.Pgrid, metadata=json.dumps(metadata))
    return cache


def render(cache, output, fps):
    with np.load(cache) as saved:
        p, u, v = saved['pressure'], saved['u'], saved['v']
        x, y, records = saved['arrow_x'], saved['arrow_y'], saved['records']
        meta = json.loads(str(saved['metadata']))
    if meta['source_sha256'] != source_hashes():
        raise RuntimeError('Cache source mismatch; rerun the simulation.')
    if not FFMpegWriter.isAvailable():
        raise RuntimeError('FFmpeg is required; simulated fields remain in the cache.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    c = NS(**meta['settings'])
    # One symmetric range for the entire movie; saturated tails are marked on the bar.
    limit = max(.1, float(np.nanpercentile(np.abs(p), 99)))
    limit = np.ceil(limit*10)/10
    clipped = np.count_nonzero(np.abs(p) > limit)/np.count_nonzero(np.isfinite(p))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
    fig.text(.08, .935, 'Flow around a NACA 2412 wing', fontsize=23, weight='bold')
    fig.text(.08, .887, f'Pressure colour + velocity arrows  |  30°  |  Re$_c$ = {meta["reynolds_based_on_chord"]:g}  |  {c.nx} × {c.ny} cells', fontsize=12, color='#475569')
    ax = fig.add_axes([.08, .17, .78, .64])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#343d49')
    im = ax.imshow(p[0].T, origin='lower', extent=(0, c.lx, 0, c.ly),
                   cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest', aspect='equal')
    q = ax.quiver(x, y, np.ma.masked_invalid(u[0].T), np.ma.masked_invalid(v[0].T),
                  angles='xy', scale_units='xy', scale=23., width=.0022, color='#20262e')
    ax.quiverkey(q, .90, 1.045, 1., 'U = 1', labelpos='E', coordinates='axes')
    ax.set(xlabel='x (solver units)', ylabel='y (solver units)', xlim=(0, c.lx), ylim=(0, c.ly))
    cb = fig.colorbar(im, cax=fig.add_axes([.89, .23, .016, .52]), extend='both')
    cb.set_label('Relative pressure  (p − p$_{inlet}$) / ρ', labelpad=12)
    label = fig.text(.08, .83, '', family='monospace', fontsize=11)
    fig.text(.08, .083, 'Colour scale and arrow lengths stay fixed throughout. Dark shape: no-slip wing.', fontsize=10, color='#475569')
    fig.text(.08, .047, 'Reference-video framing; newly specified run. Compact fixed boundaries; not a converged benchmark.', fontsize=9, color='#475569')
    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    started = time.perf_counter()
    with writer.saving(fig, str(output), dpi=100):
        for i, row in enumerate(records):
            im.set_data(p[i].T)
            q.set_UVC(np.ma.masked_invalid(u[i].T), np.ma.masked_invalid(v[i].T))
            label.set_text(f't = {row[0]:5.2f}    tU/c = {row[0]*c.velocity/meta["chord"]:5.1f}    max |div u| = {row[1]:.1e}    CFL = {row[2]:.3f}')
            writer.grab_frame()
        fig.savefig(output.with_suffix('.png'), dpi=100)
    plt.close(fig)
    meta.update(video_fps=fps, video_duration_seconds=len(p)/fps, video_codec='H.264 / yuv420p',
                pressure_colour_limits=[-limit, limit], pressure_colour_clipped_fraction=float(clipped),
                rendering_seconds=time.perf_counter()-started)
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(['time', 'max_abs_divergence', 'cfl', 'net_boundary_flux',
                             'max_blocked_face_velocity', 'pressure_reference', 'relative_pressure_min', 'relative_pressure_max'])
        csv_writer.writerows(records)
    print(f'Rendered {output} ({output.stat().st_size/1e6:.2f} MB)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=400)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--dt', type=float, default=.0005)
    parser.add_argument('--tf', type=float, default=10.)
    parser.add_argument('--re', type=float, default=600., help='Solver viscosity denominator; chord Reynolds number = 0.5 * re')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--frame-interval', type=float, default=.05)
    parser.add_argument('--cache', default=str(ROOT/'.cache/naca_pressure_velocity.npz'))
    parser.add_argument('--output', default=str(ROOT/'movies/naca_pressure_velocity_corrected.mp4'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--simulate-only', action='store_true')
    mode.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.fps <= 0 or args.frame_interval <= 0:
        parser.error('fps and frame-interval must be positive')
    cache = Path(args.cache) if args.render_only else simulate(args)
    if not args.simulate_only:
        render(cache, args.output, args.fps)


if __name__ == '__main__':
    main()
