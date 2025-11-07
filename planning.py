import numpy as np
from kinematics import JOINT_SPEED_LIMITS, JOINT_ACCEL_LIMITS, SOFT_LIMIT_SCALE

def min_jerk_trajectory(q0, q1, T, dt=0.01):
    """Quintic polynomial per joint."""
    t = np.arange(0, T+dt, dt)
    a0 = q0
    a1 = np.zeros_like(q0)
    a2 = np.zeros_like(q0)
    a3 = 10*(q1 - q0) / T**3
    a4 = -15*(q1 - q0) / T**4
    a5 = 6*(q1 - q0) / T**5
    q = []
    dq = []
    ddq = []
    for ti in t:
        q.append(a0 + a1*ti + a2*ti**2 + a3*ti**3 + a4*ti**4 + a5*ti**5)
        dq.append(a1 + 2*a2*ti + 3*a3*ti**2 + 4*a4*ti**3 + 5*a5*ti**4)
        ddq.append(2*a2 + 6*a3*ti + 12*a4*ti**2 + 20*a5*ti**3)
    return t, np.vstack(q), np.vstack(dq), np.vstack(ddq)


def rescale_duration(q, dq, ddq, dt, soft=True):
    """Stretch trajectory duration if any joint exceeds limits."""
    speed_limits = JOINT_SPEED_LIMITS * (SOFT_LIMIT_SCALE if soft else 1.0)
    accel_limits = JOINT_ACCEL_LIMITS * (SOFT_LIMIT_SCALE if soft else 1.0)
    v_max = np.max(np.abs(dq), axis=0)
    a_max = np.max(np.abs(ddq), axis=0)
    scale_v = np.max(v_max / speed_limits)
    scale_a = np.sqrt(np.max(a_max / accel_limits))
    scale = max(scale_v, scale_a, 1.0)
    return scale


def build_trajectory(q_segments, duration=2.0, dt=0.01):
    """Chain multiple segments, uniform duration per segment."""
    times = []
    q_list = []
    dq_list = []
    ddq_list = []
    total_time = 0.0
    for i in range(len(q_segments) - 1):
        t, q, dq, ddq = min_jerk_trajectory(q_segments[i], q_segments[i+1], duration, dt)
        scale = rescale_duration(q, dq, ddq, dt)
        if scale > 1.0:
            T_scaled = duration * scale
            t, q, dq, ddq = min_jerk_trajectory(q_segments[i], q_segments[i+1], T_scaled, dt)
        t += total_time
        total_time = t[-1]
        times.append(t)
        q_list.append(q)
        dq_list.append(dq)
        ddq_list.append(ddq)
    t_concat = np.concatenate(times)
    q_concat = np.vstack(q_list)
    dq_concat = np.vstack(dq_list)
    ddq_concat = np.vstack(ddq_list)
    return t_concat, q_concat, dq_concat, ddq_concat
