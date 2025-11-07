import numpy as np
from kinematics import fk, extrinsic_xyz, ik, JOINT_LIMITS
from planning import build_trajectory
from animate import animate_trajectory

HOME = np.zeros(7)

def solve_and_plan(target_pose, q0=None, waypoints=None, options=None):
    """
    target_pose: dict {
      'position': [x, y, z],
      'orientation': [theta_x_deg, theta_y_deg, theta_z_deg]
    }
    q0: initial joint angles (radians)
    """
    if q0 is None:
        q_seed = HOME.copy()
    else:
        q_seed = np.array(q0, dtype=float)

    target_p = np.array(target_pose['position'], dtype=float)
    target_R = extrinsic_xyz(target_pose['orientation'])

    q_star, ik_success, info = ik(target_p, target_R, q_seed)
    achieved_p, achieved_R = fk(q_star)
    pos_err = np.linalg.norm(achieved_p - target_p)
    ori_err = np.linalg.norm(achieved_R - target_R)

    status = {
        'ik_success': ik_success,
        'iterations': info['iterations'],
        'pos_error': pos_err,
        'ori_error': ori_err,
        'achieved_pose': {'position': achieved_p.tolist(), 'orientation_matrix': achieved_R.tolist()}
    }

    if waypoints is None:
        q_segments = [q_seed, q_star]
    else:
        q_segments = [q_seed] + waypoints + [q_star]

    duration = options.get('duration', 2.0) if options else 2.0
    t, q, dq, ddq = build_trajectory(q_segments, duration=duration)

    return {
        'trajectory': {
            'time': t,
            'position': q,
            'velocity': dq,
            'acceleration': ddq
        },
        'status': status
    }


def render_animation(result, target_pose=None, save_path=None, fps=30, interactive=True):
    times = result['trajectory']['time']
    qs = result['trajectory']['position']
    if target_pose is not None:
        p_tar = np.array(target_pose['position'])
        R_tar = extrinsic_xyz(target_pose['orientation'])
        animate_trajectory(times, qs, target_pose=(p_tar, R_tar),
                           save_path=save_path, fps=fps, interactive=interactive)
    else:
        animate_trajectory(times, qs, target_pose=None,
                           save_path=save_path, fps=fps, interactive=interactive)
