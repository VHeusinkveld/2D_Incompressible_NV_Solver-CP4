"""Compatible MAC-grid operators and diagnostics for fixed velocity boundaries."""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components


def velocity_masks(const, obj):
    if obj.sort is None:
        return (np.zeros((const.nx - 1, const.ny), dtype=bool),
                np.zeros((const.nx, const.ny - 1), dtype=bool))
    return obj.Ugrid.astype(bool), obj.Vgrid.astype(bool)


def pressure_operator(const, LP, blocked_u, blocked_v):
    """Assemble -D M G from precisely the faces used to correct velocity.

    There is one pressure gauge per connected component (including isolated
    solid cells). Fixed normal boundary velocities imply homogeneous Neumann
    pressure conditions; no dt or viscosity coefficient belongs in this matrix.
    """
    ids = np.arange(const.nx * const.ny).reshape((const.nx, const.ny), order='F')
    active_u, active_v = ~blocked_u, ~blocked_v
    left = np.r_[ids[:-1, :][active_u], ids[:, :-1][active_v]]
    right = np.r_[ids[1:, :][active_u], ids[:, 1:][active_v]]
    weight = np.r_[np.full(active_u.sum(), 1 / const.hx**2),
                   np.full(active_v.sum(), 1 / const.hy**2)]
    matrix = sparse.coo_matrix(
        (np.r_[weight, weight, -weight, -weight],
         (np.r_[left, right, left, right], np.r_[left, right, right, left])),
        shape=(ids.size, ids.size)).tocsc()
    count, labels = connected_components(matrix, directed=False)
    # np.unique gives the first cell of each component without quadratic scans.
    _, anchors = np.unique(labels, return_index=True)
    free = np.ones(ids.size)
    free[anchors] = 0
    select = sparse.diags(free, format='csc')
    LP.Lp_neumann = matrix
    LP.Lp = (select @ matrix @ select + sparse.diags(1 - free)).tocsc()
    LP.pressure_labels = labels
    LP.pressure_components = count
    LP.pressure_anchors = anchors
    return LP


def pressure_rhs(LP, divergence, dt):
    rhs = -divergence.ravel(order='F') / dt
    balance = np.bincount(LP.pressure_labels, weights=rhs,
                          minlength=LP.pressure_components)
    magnitude = np.bincount(LP.pressure_labels, weights=np.abs(rhs),
                            minlength=LP.pressure_components)
    if np.any(np.abs(balance) > 1e-10 * np.maximum(1, magnitude)):
        raise ValueError('Incompatible boundary flux: each connected fluid region '
                         'must have equal inflow and outflow.')
    rhs[LP.pressure_anchors] = 0
    return rhs


def obstacle_diffusion(matrix, blocked, tangential_axis, coefficient):
    """Zero normal wall faces; reflect tangential velocity across half-cell walls.

    For tangential velocities the ghost value is -u, so a blocked neighbour
    adds one further coefficient to the diagonal (the matrix already has one).
    """
    neighbours = np.zeros(blocked.shape, dtype=float)
    if tangential_axis == 1:
        neighbours[:, 1:] += blocked[:, :-1]
        neighbours[:, :-1] += blocked[:, 1:]
    else:
        neighbours[1:, :] += blocked[:-1, :]
        neighbours[:-1, :] += blocked[1:, :]
    active = (~blocked).ravel(order='F').astype(float)
    select = sparse.diags(active, format='csc')
    matrix = matrix + sparse.diags((coefficient * neighbours).ravel(order='F'))
    return (select @ matrix @ select + sparse.diags(1 - active)).tocsc()


def divergence(const, bc, data):
    return (np.diff(np.vstack((bc.uW, data.U, bc.uE)), axis=0) / const.hx +
            np.diff(np.vstack((bc.vS, data.V.T, bc.vN)).T, axis=1) / const.hy)


def cell_velocity(const, bc, data):
    u = np.vstack((bc.uW, data.U, bc.uE))
    v = np.vstack((bc.vS, data.V.T, bc.vN)).T
    return (u[:-1] + u[1:]) / 2, (v[:, :-1] + v[:, 1:]) / 2


def courant_number(const, bc, data):
    umax = max(np.max(np.abs(data.U)), *(np.max(np.abs(getattr(bc, key)))
                                        for key in ('uW', 'uE', 'uN', 'uS')))
    vmax = max(np.max(np.abs(data.V)), *(np.max(np.abs(getattr(bc, key)))
                                        for key in ('vW', 'vE', 'vN', 'vS')))
    return const.dt * (umax / const.hx + vmax / const.hy)
