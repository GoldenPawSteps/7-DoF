import numpy as np
from solver import solve_and_plan
from kinematics import fk, extrinsic_xyz
from animate import animate_trajectory

tests = [
    {
        "name": "Home",
        "target": {
            "position": [0.45, 0.0, 0.5],
            "orientation": [0.0, 0.0, 0.0]
        }
    },
    {
        "name": "Retract",
        "target": {
            "position": [0.3, 0.0, 0.4],
            "orientation": [0.0, 90.0, 0.0]
        }
    },
    {
        "name": "EdgeReach",
        "target": {
            "position": [0.85, 0.1, 0.3],
            "orientation": [10.0, 0.0, 30.0]
        }
    }
]

if __name__ == "__main__":
    for test in tests:
        print(f"\n--- {test['name']} ---")
        result = solve_and_plan(test['target'])
        status = result['status']
        print(f"IK success: {status['ik_success']}, iterations: {status['iterations']}")
        print(f"Position error: {status['pos_error']*1000:.3f} mm")
        print(f"Orientation residual norm: {status['ori_error']:.5f}")
        animate = False  # set True for quick preview
        if animate:
            from solver import render_animation
            render_animation(result, target_pose=test['target'], interactive=True)
