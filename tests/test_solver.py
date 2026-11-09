import copy
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve

from constants import constants
from functions import (initialisation, set_BC, set_BM, update_BC, creat_obj,
                       laplacian, laplace_obj, cholesky_decomposition, naca4)
from operators import divergence, pressure_operator, pressure_rhs, obstacle_diffusion
from simulation import simulation_step, simulation


def case(kind=None, nx=40, ny=24, boundary='horizontal_sea', cached=False,
         dt=1e-3, Re=100):
    c = constants(NS(nx=nx, ny=ny, lx=2., ly=1., dt=dt, tf=.02,
                     Re=Re, rho_arrow=10, velocity=1., cholesky=cached,
                     nsteps=1000, save_fig=False))
    o = creat_obj(c, NS(sort=kind, profile='2412', angle=30))
    lp = laplace_obj(c, laplacian(c, NS()), o)
    if cached:
        cholesky_decomposition(lp)
    bc = set_BM(c, set_BC(c, NS(), boundary))
    d = initialisation(c, NS())
    return c, o, lp, bc, d


class ProjectionTests(unittest.TestCase):
    def test_incompressibility_over_multiple_steps(self):
        for kind in (None, 'circle', 'naca'):
            for cached in (False, True):
                with self.subTest(kind=kind, cached=cached):
                    c, o, lp, bc, d = case(kind, cached=cached)
                    for _ in range(20):
                        simulation_step(c, bc, o, lp, d)
                        self.assertLess(np.max(np.abs(divergence(c, bc, d))), 2e-9)
                        if kind:
                            assert_allclose(d.U[o.Ugrid], 0, atol=0)
                            assert_allclose(d.V[o.Vgrid], 0, atol=0)

    def test_pressure_matrix_is_discrete_negative_divergence_gradient(self):
        c, o, lp, bc, d = case('naca')
        p = np.random.default_rng(8).normal(size=(c.nx, c.ny))
        d.U = np.diff(p, axis=0) / c.hx * ~o.Ugrid
        d.V = np.diff(p, axis=1) / c.hy * ~o.Vgrid
        for key in ('uW', 'uE', 'vN', 'vS'):
            getattr(bc, key)[:] = 0
        assert_allclose(lp.Lp_neumann @ p.ravel(order='F'),
                        -divergence(c, bc, d).ravel(order='F'), atol=2e-12)

    def test_pressure_matrix_independent_of_viscosity_and_timestep(self):
        a = case('circle', dt=1e-3, Re=10)[2]
        b = case('circle', dt=1e-4, Re=1000)[2]
        self.assertEqual((a.Lp - b.Lp).nnz, 0)

    def test_manufactured_gradient_projects_to_zero(self):
        c, o, lp, bc, d = case('circle', boundary='airfoil')
        p = np.sin(np.arange(c.nx)[:, None] * .3) * np.cos(np.arange(c.ny) * .4)
        d.U = np.diff(p, axis=0) / c.hx * ~o.Ugrid
        d.V = np.diff(p, axis=1) / c.hy * ~o.Vgrid
        q = spsolve(lp.Lp, pressure_rhs(lp, divergence(c, bc, d), 1))
        q = q.reshape((c.nx, c.ny), order='F')
        assert_allclose(d.U - np.diff(q, axis=0) / c.hx * ~o.Ugrid, 0, atol=1e-10)
        assert_allclose(d.V - np.diff(q, axis=1) / c.hy * ~o.Vgrid, 0, atol=1e-10)

    def test_gauges_for_disconnected_components(self):
        c, _, _, _, _ = case()
        u = np.zeros((c.nx - 1, c.ny), bool)
        v = np.zeros((c.nx, c.ny - 1), bool)
        u[c.nx // 2, :] = True
        lp = pressure_operator(c, NS(), u, v)
        self.assertEqual(lp.pressure_components, 2)
        p = spsolve(lp.Lp, np.zeros(c.nx*c.ny))
        assert_allclose(p, 0, atol=0)

    def test_incompatible_flux_rejected(self):
        c, o, lp, bc, d = case()
        bc.uE *= .8
        with self.assertRaisesRegex(ValueError, 'Incompatible boundary flux'):
            simulation_step(c, bc, o, lp, d)


class BoundaryTests(unittest.TestCase):
    def test_uniform_flow_on_unequal_cell_widths(self):
        for boundary in ('horizontal_sea', 'vertical_tube'):
            c, o, lp, bc, d = case(nx=30, ny=30, boundary=boundary)
            d.U[:] = float(boundary == 'horizontal_sea')
            d.V[:] = float(boundary == 'vertical_tube')
            expected = copy.deepcopy(d)
            for _ in range(5):
                simulation_step(c, bc, o, lp, d)
            assert_allclose(d.U, expected.U, atol=2e-13)
            assert_allclose(d.V, expected.V, atol=2e-13)

    def test_simulation_preserves_all_boundary_modes(self):
        for boundary in ('lid', 'horizontal_sea', 'horizontal_tube', 'vertical_tube', 'airfoil'):
            c, o, lp, bc, d = case(boundary=boundary, dt=.002)
            original = copy.deepcopy(bc)
            with patch('simulation.plot_system'):
                simulation(c, bc, o, lp, d)
            update_BC(c, bc, d)
            for key in ('uW', 'uE', 'uN', 'uS', 'vW', 'vE', 'vN', 'vS'):
                assert_allclose(getattr(bc, key), getattr(original, key), atol=0)
            self.assertLess(np.max(np.abs(divergence(c, bc, d))), 2e-9)

    def test_unsupported_periodic_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported boundary type'):
            case(boundary='periodic')

    def test_unsafe_advection_timestep_rejected(self):
        c, o, lp, bc, d = case()
        d.U[:] = 1000
        with self.assertRaisesRegex(ValueError, 'CFL'):
            simulation_step(c, bc, o, lp, d)

    def test_diffusion_discrete_sine_mode(self):
        c, _, lp, _, _ = case(boundary='airfoil')
        x = c.x[1:-1]
        y = (c.y[:-1] + c.y[1:]) / 2
        u = np.sin(np.pi*x[:, None]/c.lx) * np.sin(np.pi*y/c.ly)
        eigenvalue = (4*np.sin(np.pi/(2*c.nx))**2/c.hx**2 +
                      4*np.sin(np.pi/(2*c.ny))**2/c.hy**2)
        expected = u / (1+c.dt/c.Re*eigenvalue)
        actual = spsolve(lp.Lu, u.ravel(order='F')).reshape(u.shape, order='F')
        assert_allclose(actual, expected, atol=1e-14)

    def test_half_cell_no_slip_wall_linear_profile(self):
        c, _, lp, _, _ = case(nx=12, ny=12, boundary='airfoil')
        blocked = np.zeros((c.nx-1, c.ny), bool)
        blocked[:, :4] = True
        coefficient = c.dt/c.Re/c.hy**2
        A = obstacle_diffusion(lp.Lu, blocked, 1, coefficient)
        # Linear Couette flow above the wall between rows 3 and 4.
        profile = np.maximum(np.arange(c.ny) - 3.5, 0)
        u = np.tile(profile, (c.nx-1, 1))
        residual = (A @ u.ravel(order='F')).reshape(u.shape, order='F') - u
        assert_allclose(residual[1:-1, 4:-1], 0, atol=1e-14)


class GeometryTests(unittest.TestCase):
    def test_symmetric_naca_and_rotated_geometry(self):
        x, y = naca4('0012', np.linspace(0, 1, 201))
        self.assertEqual(len(x), 401)
        assert_allclose(x, x[::-1], atol=1e-14)
        assert_allclose(y, -y[::-1], atol=1e-14)
        c = case()[0]
        for profile in ('0012', '2412'):
            for angle in (0, 30, 60, 90):
                obj = creat_obj(c, NS(sort='naca', profile=profile, angle=angle))
                self.assertGreater(np.count_nonzero(obj.Pgrid), 0)

    def test_cylinder_symmetry_preserved(self):
        c, o, lp, bc, d = case('circle')
        simulation_step(c, bc, o, lp, d)
        assert_allclose(d.U, d.U[:, ::-1], atol=1e-10)
        assert_allclose(d.V, -d.V[:, ::-1], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
