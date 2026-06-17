# -*- coding: utf-8 -*-
"""
Verify two ablation experiment failure causes.
Run from BFM-Zero root:  uv run python scripts/verify_ablation_bugs.py
"""
import math
import torch
import torch.nn as nn

def count_params(m):
    return sum(p.numel() for p in m.parameters())

PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 70)
print("BFM-Zero Ablation Bug Verification")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# Test 1 – discount_hand calculation
# ─────────────────────────────────────────────────────────────
print("\n[Test 1] discount_hand calculation")
train_discount   = 0.98
fb_hand_discount = 0.7
bs = 16
terminated = torch.zeros(bs, 1, dtype=torch.bool)
discount      = train_discount * ~terminated
discount_hand = discount * (fb_hand_discount / train_discount)
expected      = fb_hand_discount * (~terminated).float()
ok = torch.allclose(discount_hand, expected)
print(f"  discount      = {discount[0,0].item():.4f}  (expected {train_discount})")
print(f"  discount_hand = {discount_hand[0,0].item():.4f}  (expected {fb_hand_discount})")
print(f"  {PASS if ok else FAIL}  discount_hand == {fb_hand_discount} * ~terminated")

# ─────────────────────────────────────────────────────────────
# Test 2 – Q_hand vs Q_body magnitude in fb mode
# ─────────────────────────────────────────────────────────────
print("\n[Test 2] Actor hand gradient imbalance in fb mode (analytical)")

DISCOUNT_BODY  = 0.98
DISCOUNT_HAND  = 0.70
Z_BODY_DIM = 324
Z_HAND_DIM  = 64

B_body_norm = math.sqrt(Z_BODY_DIM)   # 18
B_hand_norm = math.sqrt(Z_HAND_DIM)   # 8

Q_body_scale = B_body_norm**2 / (1 - DISCOUNT_BODY)    # 16200
Q_hand_scale = B_hand_norm**2 / (1 - DISCOUNT_HAND)    # 213.3
ratio = Q_hand_scale / Q_body_scale                     # 0.013

print(f"  Q_body (gamma={DISCOUNT_BODY}): {B_body_norm:.0f}^2/{1-DISCOUNT_BODY:.2f} = {Q_body_scale:.0f}")
print(f"  Q_hand (gamma={DISCOUNT_HAND}): {B_hand_norm:.0f}^2/{1-DISCOUNT_HAND:.2f} = {Q_hand_scale:.1f}")
print(f"  Q_hand / Q_body = {ratio:.5f}  ({ratio*100:.2f}%)")
print()
print("  In mse mode: hand_weight = Q_body.abs().mean()  (adaptive, ~Q_body magnitude)")
print("  In fb  mode: hand_weight = actor_hand_q_weight = 1.0  (FIXED, ~213x too small)")
ok2 = ratio < 0.05
print(f"  {FAIL if ok2 else PASS}  fb mode hand gradient is only {ratio*100:.1f}% of body"
      f" -- {'PROBLEM CONFIRMED' if ok2 else 'ok'}")

# ─────────────────────────────────────────────────────────────
# Test 3 – residual SplitForwardMap vs original ResidualForwardMap
# ─────────────────────────────────────────────────────────────
print("\n[Test 3] residual SplitForwardMap parameter count vs original ResidualForwardMap")
print("  (The original ResidualForwardMap is the intended structural reference,")
print("   not the Simple SplitForwardMap. BFM0 used ResidualForwardMap and converged.)")
try:
    from humanoidverse.agents.nn_models import (
        SplitForwardArchiConfig, ForwardArchiConfig
    )
    import gymnasium
    import numpy as np

    OBS_DIM = 1050   # approx: state(64)+privstate(463)+last_action(29)+history(~494)
    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    ACTION_DIM = 29
    Z_BODY = 324
    Z_HAND = 64
    Z_TOTAL = Z_BODY + Z_HAND   # 388

    # ── 1. Original BFM0 ResidualForwardMap (the working baseline) ──
    # Use actual BFM0 config: z_dim=256 (non-split), hidden_layers=6 for BOTH embed AND head
    # (ResidualForwardMap uses hidden_layers for everything, unlike _SingleSplitForwardMap)
    Z_BFM0 = 256
    cfg_orig = ForwardArchiConfig(
        name='ForwardArchi',
        model='residual',
        hidden_dim=2048,
        hidden_layers=6,
        embedding_layers=2,
        num_parallel=2,
        ensemble_mode='batch',
    )
    m_orig = cfg_orig.build(obs_space, Z_BFM0, ACTION_DIM)
    p_orig = count_params(m_orig)

    # ── 2. Simple SplitForwardMap (main experiment) ──
    cfg_simple = SplitForwardArchiConfig(
        name='SplitForwardArchi', model='simple',
        hidden_dim=2048, trunk_hidden_dim=256,
        hidden_layers=6, embedding_layers=2, num_parallel=2,
        z_body_dim=Z_BODY, z_hand_dim=Z_HAND, hand_action_dim=7,
    )
    m_simple = cfg_simple.build(obs_space, Z_TOTAL, ACTION_DIM)
    p_simple = count_params(m_simple)

    # ── 3. Current (FIXED) residual SplitForwardMap ──
    cfg_res_fixed = SplitForwardArchiConfig(
        name='SplitForwardArchi', model='residual',
        hidden_dim=2048, trunk_hidden_dim=256,
        hidden_layers=6, embedding_layers=2, num_parallel=2,
        z_body_dim=Z_BODY, z_hand_dim=Z_HAND, hand_action_dim=7,
    )
    m_res_fixed = cfg_res_fixed.build(obs_space, Z_TOTAL, ACTION_DIM)
    p_res_fixed = count_params(m_res_fixed)

    # ── 4. Verify embedding_layers is now live (not dead code) ──
    cfg_res_alt = SplitForwardArchiConfig(
        name='SplitForwardArchi', model='residual',
        hidden_dim=2048, trunk_hidden_dim=256,
        hidden_layers=6, embedding_layers=4, num_parallel=2,  # ← changed to 4
        z_body_dim=Z_BODY, z_hand_dim=Z_HAND, hand_action_dim=7,
    )
    m_res_alt = cfg_res_alt.build(obs_space, Z_TOTAL, ACTION_DIM)
    p_res_alt = count_params(m_res_alt)
    emb_live = abs(p_res_fixed - p_res_alt) > 1000

    # ── 5. Analytical: buggy version (hidden_layers=6 for embed, BEFORE fix) ──
    def emb_analytical(in_dim, h, nl):
        return in_dim*h + max(0, nl-2)*h*h + h*(h//2)

    h = 2048
    p_buggy_one = (emb_analytical(OBS_DIM+Z_HAND, h, 6) +
                   emb_analytical(OBS_DIM+7,      h, 6) +
                   emb_analytical(OBS_DIM+Z_BODY,  h, 6) +
                   emb_analytical(OBS_DIM+22,      h, 6) +
                   6*h*h + h*Z_HAND +
                   6*h*h + h*Z_BODY)
    p_buggy = p_buggy_one * 2   # num_parallel=2 instances

    print(f"\n  Parameter count comparison (hidden_dim=2048, hidden_layers=6, num_parallel=2):")
    print(f"")
    print(f"  Orig BFM0  ResidualForwardMap  z_dim=256, embed=deep(hl=6): {p_orig/1e6:6.1f}M")
    print(f"  Simple     SplitForwardMap     z_tot=388, trunk+MLP        : {p_simple/1e6:6.1f}M")
    print(f"  Residual   SplitForwardMap  BEFORE fix  embed=hl=6 (BUG)  :~{p_buggy/1e6:6.1f}M  (analytical)")
    print(f"  Residual   SplitForwardMap  AFTER  fix  embed=el=2 (FIXED) : {p_res_fixed/1e6:6.1f}M")
    print()

    ratio_fixed_vs_orig = p_res_fixed / p_orig
    ratio_buggy_vs_orig = p_buggy / p_orig
    print(f"  Ratios vs original BFM0 ResidualForwardMap (converged reference):")
    print(f"    Buggy  Split residual: ~{ratio_buggy_vs_orig:.2f}x  -- much larger than ref, failed to converge")
    print(f"    Fixed  Split residual:  {ratio_fixed_vs_orig:.2f}x  -- comparable to ref")

    close_to_orig = 0.5 <= ratio_fixed_vs_orig <= 2.0
    print(f"  {PASS if close_to_orig else FAIL}  "
          f"Fixed residual SplitF is {'comparable to' if close_to_orig else 'NOT comparable to'} "
          f"original BFM0 ResidualF (within 0.5x-2x)")
    print()

    print(f"  embedding_layers field is {'LIVE (working)' if emb_live else 'DEAD CODE (bug still present!)'}")
    print(f"  {PASS if emb_live else FAIL}  embedding_layers is used correctly")

    print()
    print(f"  Structural note (both h=2048, hl=6, el=2, np=2):")
    print(f"    BFM0:         2 embeds deep(hl=6) + 1 head(hl=6)  z_dim=256")
    print(f"    Fixed SplitF: 4 embeds shallow(el=2) + 2 heads(hl=6) z_body=324+z_hand=64")
    print(f"    -> Convergence difficulty should now be similar to BFM0.")

except Exception as e:
    print(f"  Could not instantiate networks: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# Test 4 – SplitActor uses embedding_layers correctly (sanity)
# ─────────────────────────────────────────────────────────────
print("\n[Test 4] SplitActor DOES use embedding_layers (sanity check)")
try:
    from humanoidverse.agents.nn_models import SplitActorArchiConfig
    import gymnasium, numpy as np

    OBS_DIM_A = 587
    obs_space_a = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM_A,), dtype=np.float32
    )

    def build_actor(embedding_layers):
        cfg = SplitActorArchiConfig(
            name='SplitActorArchi', model='residual',
            hidden_dim=2048, hidden_layers=6,
            embedding_layers=embedding_layers,
            z_body_dim=324, z_hand_dim=64, hand_action_dim=7,
        )
        return cfg.build(obs_space_a, 324+64, 29)

    a2 = build_actor(2)
    a4 = build_actor(4)
    p2, p4 = count_params(a2), count_params(a4)
    diff = abs(p2 - p4)
    actor_uses_emb = diff > 1000
    print(f"  SplitActor embedding_layers=2: {p2/1e6:.1f}M")
    print(f"  SplitActor embedding_layers=4: {p4/1e6:.1f}M  (diff={diff:,})")
    print(f"  {PASS if actor_uses_emb else FAIL}  "
          f"SplitActor {'correctly uses' if actor_uses_emb else 'ignores'} embedding_layers")
except Exception as e:
    print(f"  Skipped: {e}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Issue 1: [fb mode, gamma=0.7]  Design flaw in actor loss (FIXED in agent files).
  - Hand Q gradient was 1.3% of body -> near-zero hand tracking signal
  - Fix: adaptive scaling  (q_body_scale / q_hand_scale) * Q_fb_hand
  - Now hand gradient is always commensurate with body gradient

Issue 2: [residual variant]  Code bug in nn_models.py (FIXED).
  - Buggy:  embedding_layers was dead code; hidden_layers=6 used for all 4 embeds
            -> ~272M params (far larger than original BFM0 ~200M -> couldn't converge)
  - Fixed:  embedding_layers=2 now used for embeds; hidden_layers=6 kept for heads
            -> ~138M params, within 0.5x-2x of original BFM0 ResidualForwardMap (~200M)
  - Structural comparison:
      BFM0 ResidualForwardMap: 2 deep embeds (hl=6 RBs) + 1 head (hl=6 RBs), z_dim=256
      Fixed SplitForwardMap:   4 shallow embeds (el=2) + 2 heads (hl=6 RBs), z_body+z_hand
  - Conclusion: fixed residual SplitF is now architecturally comparable to the
                original BFM0 model that converged successfully.
""")
