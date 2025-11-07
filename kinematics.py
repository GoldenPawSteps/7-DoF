import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# DH parameters (Classical) for the 7-DoF Gen3 (meters)
DH_TABLE = np.array([
    [np.pi,       0.0,  0.0,                      0.0],                       # shim
    [np.pi/2.0,   0.0, -(0.1564+0.1284),          0.0],                       # q1
    [np.pi/2.0,   0.0, -(0.0054+0.0064),          np.pi],                     # q2
    [np.pi/2.0,   0.0, -(0.2104+0.2104),          np.pi],                     # q3
    [np.pi/2.0,   0.0, -(0.0064+0.0064),          np.pi],                     # q4
    [np.pi/2.0,   0.0, -(0.2084+0.1059),          np.pi],                     # q5
    [np.pi/2.0,   0.0,  0.0,                      np.pi],                     # q6
    [np.pi,       0.0, -(0.1059+0.0615),          np.pi],                     # q7
])

# joint hard limits (radians)
JOINT_LIMITS = np.array([
    [ -np.inf,  np.inf],          # j1
    [-128.9*DEG2RAD, 128.9*DEG2RAD],  # j2
    [ -np.inf,  np.inf],          # j3
    [-147.8*DEG2RAD, 147.8*DEG2RAD],  # j4
    [ -np.inf,  np.inf],          # j5
    [-120.3*DEG2RAD, 120.3*DEG2RAD],  # j6
    [ -np.inf,  np.inf],          # j7
])

JOINT_SPEED_LIMITS = np.deg2rad(np.array([79.64, 79.64, 79.64, 79.64, 69.91, 69.91, 69.91]))
JOINT_ACCEL_LIMITS = np.deg2rad(np.array([297.94, 297.94, 297.94, 297.94, 572.95, 572.95, 572.95]))

SOFT_LIMIT_SCALE = 0.9  # plan at 90% of hard limits


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def extrinsic_xyz(angles_deg):
    """Rx(θx) * Ry(θy) * Rz(θz) with θ in degrees."""
    ax, ay, az = np.deg2rad(angles_deg)
    return rot_x(ax) @ rot_y(ay) @ rot_z(az)

def dh_transform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0,0,0,1]
    ])


def fk(q):
    """Forward kinematics: returns (position, rotation matrix)."""
    assert len(q) == 7
    T = np.eye(4)
    for i in range(len(DH_TABLE)):
        alpha, a, d, theta_offset = DH_TABLE[i]
        theta = theta_offset + (q[i-1] if i > 0 else 0.0)
        T = T @ dh_transform(alpha, a, d, theta)
    p = T[:3,3]
    R = T[:3,:3]
    return p, R


def jacobian(q, epsilon=1e-8):
    """Geometric Jacobian via numerical differentiation."""
    p0, _ = fk(q)
    J = np.zeros((6,7))
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = epsilon
        p_eps, R_eps = fk(q + dq)
        dp = (p_eps - p0) / epsilon
        _, R = fk(q)
        dR = (R_eps - R) / epsilon
        omega = np.array([
            dR[2,1] - dR[1,2],
            dR[0,2] - dR[2,0],
            dR[1,0] - dR[0,1]
        ]) / 2.0
        J[:3, i] = dp
        J[3:, i] = omega
    return J


def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))


def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def project_to_limits(q):
    q_proj = []
    for i, qi in enumerate(q):
        lo, hi = JOINT_LIMITS[i]
        if np.isfinite(lo) and qi < lo:
            q_proj.append(lo)
        elif np.isfinite(hi) and qi > hi:
            q_proj.append(hi)
        else:
            q_proj.append(qi)
    return np.array(q_proj)


def joint_limit_penalty(q):
    penalty = 0.0
    grad = np.zeros_like(q)
    for i, qi in enumerate(q):
        lo, hi = JOINT_LIMITS[i]
        span = hi - lo if np.isfinite(hi - lo) else np.inf
        if not np.isfinite(span):
            continue
        mid = (lo + hi) / 2.0
        dist = (qi - mid) / (span / 2.0)
        penalty += dist**2
        grad[i] = 2 * dist / (span / 2.0)
    return penalty, grad


def ik(target_p, target_R, q_seed, max_iters=200, tol_pos=1e-4, tol_ori=1e-3, damping_base=1e-4):
    """
    Damped Least Squares IK with null-space projection.
    Returns (q_star, status, info_dict)
    """
    q = q_seed.copy()
    success = False
    info = {'iterations': 0, 'final_error': None}
    for it in range(max_iters):
        p, R = fk(q)
        err_pos = target_p - p
        dR = target_R @ R.T
        ori_error = np.array([
            dR[2,1] - dR[1,2],
            dR[0,2] - dR[2,0],
            dR[1,0] - dR[0,1]
        ]) / 2.0
        err = np.concatenate([err_pos, ori_error])
        if np.linalg.norm(err_pos) < tol_pos and np.linalg.norm(ori_error) < tol_ori:
            success = True
            break
        J = jacobian(q)
        mu = manipulability(J)
        lam = damping_base + (1.0 - min(mu, 1.0))**2
        JT = J.T
        dq_primary = JT @ np.linalg.solve(J @ JT + lam*np.eye(6), err)
        penalty, grad_penalty = joint_limit_penalty(q)
        manipulability_gradient = np.zeros_like(q)  # optional custom gradient
        dq_null = -(grad_penalty + 0.01 * manipulability_gradient)
        N = np.eye(7) - JT @ np.linalg.solve(J @ JT + lam*np.eye(6), J)
        dq = dq_primary + N @ dq_null
        q = project_to_limits(q + dq)
        info['iterations'] = it + 1
    info['final_error'] = err
    return q, success, info
