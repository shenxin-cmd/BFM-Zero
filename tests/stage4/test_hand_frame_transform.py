import torch

from humanoidverse.agents.stage4 import (
    heading_to_world_frame,
    rotate_position_jacobian_world_to_heading,
    rotate_vectors_heading_to_world_frame,
    rotate_vectors_world_to_heading_frame,
    world_to_heading_frame,
)
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


def test_heading_vector_rotation_does_not_apply_root_translation():
    dtype = torch.float64
    yaw = torch.tensor([torch.pi / 2], dtype=dtype)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
    root_quat = quat_from_angle_axis(yaw, axis, w_last=True)
    vectors_world = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=dtype)

    vectors_heading = rotate_vectors_world_to_heading_frame(vectors_world, root_quat_xyzw=root_quat)
    restored = rotate_vectors_heading_to_world_frame(vectors_heading, root_quat_xyzw=root_quat)

    assert torch.allclose(restored, vectors_world, atol=1e-10)


def test_position_jacobian_rotates_each_column_as_a_vector():
    dtype = torch.float64
    yaw = torch.tensor([torch.pi / 2], dtype=dtype)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
    root_quat = quat_from_angle_axis(yaw, axis, w_last=True)
    jac_world = torch.zeros(1, 3, 4, dtype=dtype)
    jac_world[:, :, 0] = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype)
    jac_world[:, :, 1] = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)

    jac_heading = rotate_position_jacobian_world_to_heading(jac_world, root_quat_xyzw=root_quat)

    assert jac_heading.shape == (1, 3, 4)
    assert torch.allclose(jac_heading[:, :, 0], torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype), atol=1e-10)
    assert torch.allclose(jac_heading[:, :, 1], torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype), atol=1e-10)
