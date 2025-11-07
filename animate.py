import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from kinematics import fk, rot_x, rot_y, rot_z

LINK_COLOR = '#2E86AB'
JOINT_COLOR = '#ED553B'
TARGET_COLOR = '#4FB99F'


def compute_link_frames(q):
    frames = []
    T = np.eye(4)
    frames.append(T.copy())
    from kinematics import DH_TABLE, dh_transform
    for i in range(len(DH_TABLE)):
        alpha, a, d, theta_offset = DH_TABLE[i]
        theta = theta_offset + (q[i-1] if i > 0 else 0.0)
        T = T @ dh_transform(alpha, a, d, theta)
        frames.append(T.copy())
    return frames


def plot_frame(ax, T, length=0.05):
    p = T[:3,3]
    axes = T[:3,:3]
    ax.quiver(p[0], p[1], p[2], axes[0,0]*length, axes[1,0]*length, axes[2,0]*length,
              color='r', linewidth=2)
    ax.quiver(p[0], p[1], p[2], axes[0,1]*length, axes[1,1]*length, axes[2,1]*length,
              color='g', linewidth=2)
    ax.quiver(p[0], p[1], p[2], axes[0,2]*length, axes[1,2]*length, axes[2,2]*length,
              color='b', linewidth=2)


def animate_trajectory(times, qs, target_pose=None, save_path=None, fps=30, interactive=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-0.5, 0.9])
    ax.set_ylim3d([-0.6, 0.6])
    ax.set_zlim3d([0.0, 1.2])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Kinova Gen3 Motion')

    if target_pose is not None:
        p_tar, R_tar = target_pose
        T_tar = np.eye(4)
        T_tar[:3,:3] = R_tar
        T_tar[:3,3] = p_tar
        plot_frame(ax, T_tar, length=0.08)

    link_lines, = ax.plot([], [], [], '-o', color=LINK_COLOR, lw=2, markersize=4)

    def init():
        link_lines.set_data([], [])
        link_lines.set_3d_properties([])
        return link_lines,

    def update(frame):
        q = qs[frame]
        frames = compute_link_frames(q)
        xs = [Tf[0,3] for Tf in frames]
        ys = [Tf[1,3] for Tf in frames]
        zs = [Tf[2,3] for Tf in frames]
        link_lines.set_data(xs, ys)
        link_lines.set_3d_properties(zs)
        return link_lines,

    ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init,
                                  interval=1000.0/fps, blit=True)

    if save_path:
        Writer = animation.writers['ffmpeg']
        ani.save(save_path, writer=Writer(fps=fps, codec='libx264'))
        print(f'[INFO] Animation saved to {save_path}')

    if interactive:
        plt.show()

    plt.close(fig)
