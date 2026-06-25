import torch

from humanoidverse.agents.stage4 import heading_to_world_frame, world_to_heading_frame
from humanoidverse.utils.torch_utils import quat_from_angle_axis


def test_world_heading_world_roundtrip_for_points():
    dtype = torch.float64
    root_pos = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 1.5]], dtype=dtype)
    yaw = torch.tensor([0.7, -1.2], dtype=dtype)
    axis = torch.zeros(2, 3, dtype=dtype)
    axis[:, 2] = 1.0
    root_quat = quat_from_angle_axis(yaw, axis, w_last=True)
    points_world = torch.tensor(
        [
            [[1.5, -1.0, 0.75], [0.5, -2.5, 0.0]],
            [[-0.25, 1.0, 2.0], [-1.0, 0.5, 1.25]],
        ],
        dtype=dtype,
    )

    points_heading = world_to_heading_frame(
        points_world,
        root_pos_world=root_pos,
        root_quat_xyzw=root_quat,
    )
    reconstructed = heading_to_world_frame(
        points_heading,
        root_pos_world=root_pos,
        root_quat_xyzw=root_quat,
    )

    assert torch.allclose(reconstructed, points_world, atol=1e-10)


def test_world_to_heading_frame_uses_heading_not_full_roll_pitch():
    dtype = torch.float64
    root_pos = torch.zeros(1, 3, dtype=dtype)
    yaw = torch.tensor([torch.pi / 2], dtype=dtype)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
    root_quat = quat_from_angle_axis(yaw, axis, w_last=True)
    point_world = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype)

    point_heading = world_to_heading_frame(
        point_world,
        root_pos_world=root_pos,
        root_quat_xyzw=root_quat,
    )

    assert torch.allclose(point_heading, torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype), atol=1e-10)
