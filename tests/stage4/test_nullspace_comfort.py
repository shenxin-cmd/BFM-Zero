import torch

from humanoidverse.agents.stage4 import comfortable_posture_nullspace_delta, nullspace_projector


def test_nullspace_projector_preserves_primary_task_for_full_row_rank_jacobian():
    jac = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
            ]
        ]
    )

    projector = nullspace_projector(jac, damping=0.0)

    assert projector.shape == (1, 4, 4)
    assert torch.allclose(jac @ projector, torch.zeros(1, 3, 4), atol=1e-5)


def test_comfortable_posture_nullspace_delta_keeps_task_motion_near_zero():
    jac = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
            ]
        ]
    )
    q = torch.tensor([[0.3, -0.4, 0.2, 0.1]])
    comfortable_q = torch.zeros_like(q)

    dq_null = comfortable_posture_nullspace_delta(
        jac,
        q,
        comfortable_q,
        damping=0.0,
        gain=1.0,
        max_joint_delta=1.0,
    )

    task_motion = (jac @ dq_null.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(task_motion, torch.zeros_like(task_motion), atol=1e-5)
    assert torch.linalg.norm(q + dq_null - comfortable_q) < torch.linalg.norm(q - comfortable_q)
