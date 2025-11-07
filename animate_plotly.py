import numpy as np
import plotly.graph_objects as go
from typing import Optional, Sequence, Dict, Any

from kinematics import DH_TABLE, dh_transform


def _joint_positions(q: np.ndarray) -> np.ndarray:
    """
    Compute the sequence of joint frame origins (including base and tool).
    Returns an (N, 3) array of positions in meters.
    """
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    for i in range(len(DH_TABLE)):
        alpha, a, d, theta_offset = DH_TABLE[i]
        theta = theta_offset + (q[i - 1] if i > 0 else 0.0)
        T = T @ dh_transform(alpha, a, d, theta)
        positions.append(T[:3, 3].copy())
    return np.vstack(positions)


def render_plotly_trajectory(
    times: np.ndarray,
    qs: np.ndarray,
    target_pose: Optional[Dict[str, Sequence[float]]] = None,
    output_html: Optional[str] = None,
    sample_step: int = 1,
    show: bool = True,
    line_color: str = "#2E86AB",
    joint_color: str = "#ED553B",
    target_color: str = "#4FB99F",
    axis_limits: Optional[Dict[str, Sequence[float]]] = None,
) -> go.Figure:
    """
    Build an interactive Plotly animation for the Kinova Gen3 trajectory.

    Parameters
    ----------
    times : (F,) array
        Time stamps (seconds).
    qs : (F, 7) array
        Joint angles (radians) per frame.
    target_pose : dict, optional
        {"position": [x, y, z], "orientation": [deg_x, deg_y, deg_z]}.
        Orientation is not visualized (position marker only).
    output_html : str, optional
        Path to write an interactive HTML file.
    sample_step : int, optional
        Downsample factor for frames (use >1 for long trajectories).
    show : bool, optional
        Call `fig.show()` for inline rendering (True in notebooks).
    line_color, joint_color, target_color : str
        Hex colors for links, joints, and the target marker.
    axis_limits : dict, optional
        {"x": [xmin, xmax], "y": [...], "z": [...]} overrides default cube.

    Returns
    -------
    go.Figure
        The Plotly figure handle (can be further customized).
    """
    qs = np.asarray(qs)
    times = np.asarray(times)

    if qs.shape[0] != times.shape[0]:
        raise ValueError("times and qs must share the same frame count.")

    if sample_step < 1:
        raise ValueError("sample_step must be >= 1.")

    frame_indices = np.arange(0, len(times), sample_step)
    if frame_indices[-1] != len(times) - 1:
        frame_indices = np.append(frame_indices, len(times) - 1)

    positions_list = [_joint_positions(qs[i]) for i in frame_indices]

    # Initial pose (first frame)
    x0, y0, z0 = positions_list[0].T
    link_trace = go.Scatter3d(
        x=x0,
        y=y0,
        z=z0,
        mode="lines+markers",
        line=dict(width=6, color=line_color),
        marker=dict(size=4, color=joint_color),
        name="Arm",
    )

    data = [link_trace]

    if target_pose is not None:
        p_tar = np.array(target_pose["position"], dtype=float)
        target_trace = go.Scatter3d(
            x=[p_tar[0]],
            y=[p_tar[1]],
            z=[p_tar[2]],
            mode="markers",
            marker=dict(size=6, color=target_color, symbol="diamond"),
            name="Target",
        )
        data.append(target_trace)
    else:
        target_trace = None

    frames = []
    for k, idx in enumerate(frame_indices):
        pos = positions_list[k]
        x, y, z = pos.T
        frame_data = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                line=dict(width=6, color=line_color),
                marker=dict(size=4, color=joint_color),
                name="Arm",
            )
        ]
        if target_trace is not None:
            frame_data.append(target_trace)

        frame = go.Frame(
            data=frame_data,
            name=str(k),
            traces=list(range(len(frame_data))),
        )
        frames.append(frame)

    if axis_limits is None:
        axis_limits = {
            "x": [-0.5, 0.9],
            "y": [-0.6, 0.6],
            "z": [0.0, 1.2],
        }

    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title="Kinova Gen3 Trajectory (Interactive)",
            scene=dict(
                xaxis=dict(range=axis_limits["x"], title="X [m]"),
                yaxis=dict(range=axis_limits["y"], title="Y [m]"),
                zaxis=dict(range=axis_limits["z"], title="Z [m]"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=1000 * (times[frame_indices[1]] - times[frame_indices[0]])),
                                    mode="immediate",
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="t = ", suffix=" s"),
                    pad=dict(t=30),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{times[idx]:.2f}",
                        )
                        for i, idx in enumerate(frame_indices)
                    ],
                )
            ],
        ),
    )

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn", auto_play=False)

    if show:
        fig.show()

    return fig
