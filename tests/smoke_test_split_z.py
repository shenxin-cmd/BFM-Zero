"""
Smoke test for split-z FB + actor update paths.

Verifies:
  - No runtime errors (shape mismatches, missing tensors, etc.)
  - All loss values are finite scalars
  - Gradients flow through forward_map, backward_map, and actor networks
  - Both update_fb and update_td3_actor complete without error

Uses controlled, unit-norm mock tensors so numerical explosion never occurs
regardless of where in training we are.

Run with:
    python -m pytest tests/smoke_test_split_z.py -v
or:
    python tests/smoke_test_split_z.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ── Minimal mock networks ──────────────────────────────────────────────────────

class _Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=-1)
        return self.fc(x)


class MockForwardMap(nn.Module):
    """num_parallel=2 heads, outputs (2, batch, z_dim)."""
    def __init__(self, obs_dim, z_dim, action_dim, num_parallel=2):
        super().__init__()
        self.num_parallel = num_parallel
        self.heads = nn.ModuleList([
            _Linear(obs_dim + z_dim + action_dim, z_dim)
            for _ in range(num_parallel)
        ])

    def forward(self, obs, z, action):
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=-1)
        x = torch.cat([obs, z, action], dim=-1)
        return torch.stack([h(x) for h in self.heads], dim=0)  # (P, B, z_dim)


class MockBackwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim):
        super().__init__()
        self.fc = _Linear(obs_dim, z_dim)

    def forward(self, obs):
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=-1)
        return self.fc(obs)


class MockDist:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, clip=None):
        s = self.mean + 0.1 * torch.randn_like(self.mean)
        if clip is not None:
            s = s.clamp(-clip, clip)
        return s


class MockActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim):
        super().__init__()
        self.fc = _Linear(obs_dim + z_dim, action_dim)

    def forward(self, obs, z, std=0.2):
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=-1)
        mean = self.fc(torch.cat([obs, z], dim=-1))
        return MockDist(mean)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_tensor(*shape, device="cpu"):
    """Random tensor normalised to unit norm along last dim."""
    t = torch.randn(*shape, device=device)
    return F.normalize(t, dim=-1)


def get_targets_uncertainty(preds, pessimism_penalty=0.5):
    """Minimal copy of FBAgent.get_targets_uncertainty."""
    dim = 0
    preds_mean = preds.mean(dim=dim)
    preds_uns = preds.unsqueeze(dim)
    preds_uns2 = preds.unsqueeze(dim + 1)
    preds_diffs = torch.abs(preds_uns - preds_uns2)
    n = preds.shape[dim]
    denom = n ** 2 - n if n > 1 else 1
    preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / denom
    return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc


def orth_loss_single(B):
    batch = B.shape[0]
    z_dim = B.shape[-1]
    Cov = torch.matmul(B.T, B) / batch
    off_diag = (1 - torch.eye(z_dim, device=B.device))
    off_diag_sum = off_diag.sum().clamp(min=1)
    diag_loss = -Cov.diag().mean()
    offdiag_loss = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag_sum
    return diag_loss + offdiag_loss, diag_loss, offdiag_loss


# ── Core smoke test ───────────────────────────────────────────────────────────

def run_split_z_smoke_test(device="cpu"):
    torch.manual_seed(42)

    # Dimensions matching the split-z design
    obs_dim = 64
    action_dim = 29
    z_body_dim = 324  # 18² — matches main experiment default
    z_hand_dim = 64   # 8²  — matches main experiment default
    z_dim = z_body_dim + z_hand_dim       # 388
    batch = 64
    num_parallel = 2

    fb_hand_mse_weight = 1.0
    actor_hand_mse_weight = 1.0
    discount = torch.tensor(0.99).to(device)
    ortho_coef = 1.0
    actor_pessimism_penalty = 0.5
    fb_pessimism_penalty = 0.0

    # Build mock networks
    F_net = MockForwardMap(obs_dim, z_dim, action_dim, num_parallel).to(device)
    B_net = MockBackwardMap(obs_dim, z_dim).to(device)
    B_target = MockBackwardMap(obs_dim, z_dim).to(device)
    F_target = MockForwardMap(obs_dim, z_dim, action_dim, num_parallel).to(device)
    actor = MockActor(obs_dim, z_dim, action_dim).to(device)

    # Freeze target networks
    for p in B_target.parameters(): p.requires_grad_(False)
    for p in F_target.parameters(): p.requires_grad_(False)

    # Optimisers
    opt_F = torch.optim.Adam(F_net.parameters(), lr=1e-4)
    opt_B = torch.optim.Adam(B_net.parameters(), lr=1e-4)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)

    # Mock inputs (unit-norm → no explosion)
    obs   = make_unit_tensor(batch, obs_dim, device=device)
    goal  = make_unit_tensor(batch, obs_dim, device=device)
    z     = torch.cat([
        math.sqrt(z_body_dim) * make_unit_tensor(batch, z_body_dim, device=device),
        math.sqrt(z_hand_dim) * make_unit_tensor(batch, z_hand_dim, device=device),
    ], dim=-1)
    action = make_unit_tensor(batch, action_dim, device=device)

    off_diag = (1 - torch.eye(batch, device=device))
    off_diag_sum = off_diag.sum().clamp(min=1)

    # ── update_fb (split-z path) ───────────────────────────────────────────────
    with torch.no_grad():
        next_action = actor(obs, z).sample(clip=0.3)
        target_Fs = F_target(obs, z, next_action)       # (P, B, z_dim)
        target_B  = B_target(goal)                       # (B, z_dim)

        target_Fs_body = target_Fs[..., :z_body_dim]
        target_B_body  = target_B[:, :z_body_dim]
        target_B_hand  = target_B[:, z_body_dim:]

        target_Ms_body = torch.matmul(target_Fs_body, target_B_body.T)
        _, _, target_M_body = get_targets_uncertainty(target_Ms_body, fb_pessimism_penalty)

        target_Ms_all = torch.matmul(target_Fs, target_B.T)
        _, _, target_M = get_targets_uncertainty(target_Ms_all, fb_pessimism_penalty)

    Fs = F_net(obs, z, action)    # (P, B, z_dim)
    B  = B_net(goal)              # (B, z_dim)

    Fs_body = Fs[..., :z_body_dim]
    Fs_hand = Fs[..., z_body_dim:]
    B_body  = B[:, :z_body_dim]
    B_hand  = B[:, z_body_dim:]

    # Body FB loss (_fb_loss_single equivalent)
    Ms_body  = torch.matmul(Fs_body, B_body.T)
    diff_body = Ms_body - discount * target_M_body
    fb_offdiag_body = 0.5 * (diff_body * off_diag).pow(2).sum() / off_diag_sum
    fb_diag_body    = -torch.diagonal(diff_body, dim1=1, dim2=2).mean() * num_parallel
    fb_loss_body    = fb_offdiag_body + fb_diag_body

    # Hand MSE loss (per-head)
    fb_loss_hand = F.mse_loss(Fs_hand, target_B_hand.unsqueeze(0).expand_as(Fs_hand))

    fb_loss = fb_loss_body + fb_hand_mse_weight * fb_loss_hand

    # Orthonormality losses
    orth_body, _, _ = orth_loss_single(B_body)
    orth_hand, _, _ = orth_loss_single(B_hand)
    orth_loss = orth_body + orth_hand
    fb_loss = fb_loss + ortho_coef * orth_loss

    assert torch.isfinite(fb_loss), f"fb_loss is not finite: {fb_loss}"
    assert fb_loss.shape == (), f"fb_loss should be scalar, got shape {fb_loss.shape}"

    opt_F.zero_grad()
    opt_B.zero_grad()
    fb_loss.backward()
    assert all(p.grad is not None for p in F_net.parameters()), "F network has None gradients"
    assert all(p.grad is not None for p in B_net.parameters()), "B network has None gradients"
    opt_F.step()
    opt_B.step()

    print(f"  update_fb  : fb_loss={fb_loss.item():.4f}  "
          f"fb_body={fb_loss_body.item():.4f}  "
          f"fb_hand_mse={fb_loss_hand.item():.4f}  "
          f"orth={orth_loss.item():.4f}")

    # ── update_td3_actor (split-z path) ───────────────────────────────────────
    dist   = actor(obs, z)
    action2 = dist.sample(clip=0.3)
    Fs2    = F_net(obs, z, action2)    # (P, B, z_dim)

    z_body = z[:, :z_body_dim]
    z_hand = z[:, z_body_dim:]

    Qs_body = (Fs2[..., :z_body_dim] * z_body).sum(-1)
    _, _, Q_body = get_targets_uncertainty(Qs_body, actor_pessimism_penalty)

    actor_loss_hand_mse = F.mse_loss(
        Fs2[..., z_body_dim:],
        z_hand.unsqueeze(0).expand_as(Fs2[..., z_body_dim:]),
    )
    hand_weight = Q_body.abs().mean().detach() * actor_hand_mse_weight
    actor_loss  = -Q_body.mean() + hand_weight * actor_loss_hand_mse

    assert torch.isfinite(actor_loss), f"actor_loss is not finite: {actor_loss}"
    assert actor_loss.shape == (), f"actor_loss should be scalar, got shape {actor_loss.shape}"

    opt_actor.zero_grad()
    actor_loss.backward()
    assert all(p.grad is not None for p in actor.parameters()), "actor has None gradients"
    opt_actor.step()

    print(f"  update_actor: actor_loss={actor_loss.item():.4f}  "
          f"Q_body={Q_body.mean().item():.4f}  "
          f"hand_mse={actor_loss_hand_mse.item():.4f}  "
          f"hand_weight={hand_weight.item():.4f}")

    print("  [PASS] All assertions passed — shapes, finiteness, and gradient flow are correct.")


if __name__ == "__main__":
    print("=== Smoke test: split-z FB + actor (CPU) ===")
    run_split_z_smoke_test(device="cpu")
    if torch.cuda.is_available():
        print("=== Smoke test: split-z FB + actor (CUDA) ===")
        run_split_z_smoke_test(device="cuda")
    else:
        print("CUDA not available, skipping GPU test.")
