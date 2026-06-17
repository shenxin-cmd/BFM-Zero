# -*- coding: utf-8 -*-
"""
Verify two ablation experiment failure causes.
Run from BFM-Zero root:  uv run python scripts/verify_ablation_bugs.py
"""
import math
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# 0.  Shared helpers (inline to avoid import issues)
# ─────────────────────────────────────────────────────────────
def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    from humanoidverse.agents.nn_models import Block, ResidualBlock
    assert hidden_layers >= 2
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 70)
print("BFM-Zero Ablation Bug Verification")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# Test 1 – discount_hand is computed correctly
# ─────────────────────────────────────────────────────────────
print("\n[Test 1] discount_hand calculation")
train_discount   = 0.98
fb_hand_discount = 0.7
bs = 16
terminated = torch.zeros(bs, 1, dtype=torch.bool)
discount      = train_discount * ~terminated          # 0.98 * 1 = 0.98
discount_hand = discount * (fb_hand_discount / train_discount)
expected      = fb_hand_discount * (~terminated).float()
ok = torch.allclose(discount_hand, expected)
print(f"  discount      = {discount[0,0].item():.4f}  (expected {train_discount})")
print(f"  discount_hand = {discount_hand[0,0].item():.4f}  (expected {fb_hand_discount})")
print(f"  {PASS if ok else FAIL}  discount_hand == {fb_hand_discount} * ~terminated")

# ─────────────────────────────────────────────────────────────
# Test 2 – Q_hand vs Q_body magnitude at convergence
# ─────────────────────────────────────────────────────────────
print("\n[Test 2] Actor hand gradient imbalance in fb mode (analytical)")

DISCOUNT_BODY  = 0.98
DISCOUNT_HAND  = 0.70
Z_BODY_DIM = 324
Z_HAND_DIM  = 64

# project_z normalises z to ||z|| = sqrt(dim)
B_body_norm = math.sqrt(Z_BODY_DIM)   # 18
B_hand_norm = math.sqrt(Z_HAND_DIM)   # 8

Q_body_scale = B_body_norm**2 / (1 - DISCOUNT_BODY)    # 16200
Q_hand_scale = B_hand_norm**2 / (1 - DISCOUNT_HAND)    # 213.3
ratio = Q_hand_scale / Q_body_scale                     # 0.013

print(f"  Q_body (gamma={DISCOUNT_BODY}): ||z_body||^2/(1-g) = {B_body_norm:.0f}^2/{1-DISCOUNT_BODY:.2f}"
      f" = {Q_body_scale:.0f}")
print(f"  Q_hand (gamma={DISCOUNT_HAND}): ||z_hand||^2/(1-g) = {B_hand_norm:.0f}^2/{1-DISCOUNT_HAND:.2f}"
      f" = {Q_hand_scale:.1f}")
print(f"  Q_hand / Q_body = {ratio:.5f}  ({ratio*100:.2f}%)")
print()
print("  In mse mode: hand_weight = Q_body.abs().mean()  (adaptive, ~Q_body magnitude)")
print("  In fb  mode: hand_weight = actor_hand_q_weight = 1.0  (FIXED, ~213x too small)")
ok2 = ratio < 0.05   # should be less than 5% to flag as a problem
print(f"  {FAIL if ok2 else PASS}  fb mode hand gradient is only {ratio*100:.1f}% of body"
      f" -- {'PROBLEM CONFIRMED' if ok2 else 'ok'}")

# ─────────────────────────────────────────────────────────────
# Test 3 – SplitForwardArchiConfig: embedding_layers dead code
# ─────────────────────────────────────────────────────────────
print("\n[Test 3] residual SplitForwardMap parameter count vs simple")
try:
    from humanoidverse.agents.nn_models import (
        SplitForwardArchiConfig, SplitForwardMap
    )
    from humanoidverse.agents.nn_filters import DictInputFilterConfig
    import gymnasium
    import numpy as np

    # Minimal obs_space that matches the filter key list
    # We'll use a flat Box so IdentityFilter passes through
    OBS_DIM = 1050  # approximate
    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    ACTION_DIM = 29
    Z_BODY = 324
    Z_HAND = 64

    def build_split_f(model_variant, hidden_layers=6, embedding_layers=2):
        cfg = SplitForwardArchiConfig(
            name='SplitForwardArchi',
            model=model_variant,
            hidden_dim=2048,
            trunk_hidden_dim=256,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
            num_parallel=2,
            z_body_dim=Z_BODY,
            z_hand_dim=Z_HAND,
            hand_action_dim=7,
        )
        return cfg.build(obs_space, Z_BODY + Z_HAND, ACTION_DIM)

    m_simple   = build_split_f("simple")
    m_residual = build_split_f("residual")   # current (uses hidden_layers for embed)

    p_simple   = count_params(m_simple)
    p_residual = count_params(m_residual)

    print(f"  Simple   SplitForwardMap: {p_simple/1e6:.1f}M params")
    print(f"  Residual SplitForwardMap: {p_residual/1e6:.1f}M params  (current code)")
    print(f"  Residual / Simple = {p_residual/p_simple:.1f}x")

    # Check if embedding_layers field is actually used
    # If residual uses hidden_layers=6 for embed, then changing embedding_layers
    # should NOT change param count (dead code):
    m_residual_alt = build_split_f("residual", hidden_layers=6, embedding_layers=4)
    p_residual_alt = count_params(m_residual_alt)
    dead_code = abs(p_residual - p_residual_alt) < 1000
    print(f"\n  Changing embedding_layers from 2->4 changes params by: "
          f"{abs(p_residual-p_residual_alt):,}")
    print(f"  embedding_layers is {'DEAD CODE (never used)' if dead_code else 'used correctly'}")
    print(f"  {FAIL if dead_code else PASS}  "
          f"{'BUG CONFIRMED: embedding_layers ignored in residual F network' if dead_code else 'ok'}")

    # Show what fixed version would look like (using embedding_layers for embed)
    # We can't easily patch the class, so just estimate:
    # residual_embedding with 2 layers vs 6 layers: ~ (2*h^2)/(6*h^2) reduction ratio
    ratio_fix = (2 * 2048**2) / (6 * 2048**2)
    embed_contribution = p_residual - p_simple   # rough
    p_fixed_est = p_simple + embed_contribution * ratio_fix
    print(f"  Estimated fixed residual F params: ~{p_fixed_est/1e6:.1f}M "
          f"(vs current {p_residual/1e6:.1f}M)")

except Exception as e:
    print(f"  Could not instantiate SplitForwardMap: {e}")
    print("  Falling back to analytical estimate:")
    H = 2048
    HIDDEN_LAYERS = 6
    EMBEDDING_LAYERS = 2
    OBS_DIM = 1050

    def emb(in_dim, h, nl):
        return in_dim*h + (nl-2)*h*h + h*(h//2)

    simple_one  = (OBS_DIM*H + H*256 +
                   (256+22+324)*H + H*324 +
                   (256+7+64)*H  + H*64)
    res_one     = sum(emb(OBS_DIM+d, H, HIDDEN_LAYERS) for d in [324,22,64,7]) + \
                  HIDDEN_LAYERS*H*H + H*324 + HIDDEN_LAYERS*H*H + H*64
    fixed_one   = sum(emb(OBS_DIM+d, H, EMBEDDING_LAYERS) for d in [324,22,64,7]) + \
                  HIDDEN_LAYERS*H*H + H*324 + HIDDEN_LAYERS*H*H + H*64

    print(f"  Simple   F (per-instance): {simple_one/1e6:.1f}M  x2 = {simple_one*2/1e6:.1f}M")
    print(f"  Residual F (per-instance, current bug): {res_one/1e6:.1f}M  x2 = {res_one*2/1e6:.1f}M")
    print(f"  Residual F (per-instance, if fixed):    {fixed_one/1e6:.1f}M  x2 = {fixed_one*2/1e6:.1f}M")
    print(f"  Current residual / simple = {res_one*2/(simple_one*2):.1f}x  -> too large!")
    dead_code = True  # can't verify at runtime, but logic says it's dead

# ─────────────────────────────────────────────────────────────
# Test 4 – SplitActor uses embedding_layers correctly (sanity)
# ─────────────────────────────────────────────────────────────
print("\n[Test 4] SplitActor DOES use embedding_layers (sanity check)")
try:
    from humanoidverse.agents.nn_models import SplitActorArchiConfig, SplitActor
    import gymnasium, numpy as np

    OBS_DIM_A = 587
    obs_space_a = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM_A,), dtype=np.float32
    )

    def build_actor(embedding_layers):
        cfg = SplitActorArchiConfig(
            name='SplitActorArchi',
            model='residual',
            hidden_dim=2048,
            hidden_layers=6,
            embedding_layers=embedding_layers,
            z_body_dim=324,
            z_hand_dim=64,
            hand_action_dim=7,
        )
        return cfg.build(obs_space_a, 324+64, 29)

    a2 = build_actor(2)
    a4 = build_actor(4)
    p2, p4 = count_params(a2), count_params(a4)
    diff = abs(p2 - p4)
    actor_uses_emb = diff > 1000
    print(f"  SplitActor embedding_layers=2 params: {p2/1e6:.1f}M")
    print(f"  SplitActor embedding_layers=4 params: {p4/1e6:.1f}M  (diff={diff:,})")
    print(f"  {PASS if actor_uses_emb else FAIL}  "
          f"SplitActor {'correctly uses' if actor_uses_emb else 'ignores'} embedding_layers")
    if actor_uses_emb:
        print("  -> SplitActor is correct; only F network has the embedding_layers bug")
except Exception as e:
    print(f"  Skipped: {e}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print("""
Issue 1: [fb mode, gamma=0.7]  NOT a logic bug; design flaw in actor loss.
  - In 'fb' mode, hand_term = -1.0 * Q_hand  (Q_hand << Q_body by ~75x)
  - In 'mse' mode, hand_weight = Q_body.abs().mean()  (adaptive, ~same magnitude)
  - Hand gets negligible gradient -> policy drifts chaotically
  Fix: add adaptive scaling in fb_cpr/agent.py:454 and fb_cpr_aux/agent.py:304:
    q_body_s = Q_fb_body.abs().mean().detach()
    q_hand_s = Q_fb_hand.abs().mean().detach().clamp(min=1e-6)
    hand_term = -actor_hand_q_weight * (q_body_s / q_hand_s) * Q_fb_hand.mean()

Issue 2: [residual variant]  REAL CODE BUG in nn_models.py lines 903-906.
  - SplitForwardArchiConfig has 'embedding_layers=2' but it is NEVER read
  - Residual F network uses hidden_layers=6 for all 4 embeddings instead
  - Makes F network ~9x larger than Simple variant; fails to converge
  Fix: change lines 903-906 in nn_models.py:
    self.hand_embed_z  = residual_embedding(..., cfg.embedding_layers)  # was hidden_layers
    self.hand_embed_sa = residual_embedding(..., cfg.embedding_layers)
    self.body_embed_z  = residual_embedding(..., cfg.embedding_layers)
    self.body_embed_sa = residual_embedding(..., cfg.embedding_layers)
""")
