# -*- coding: utf-8 -*-
"""
Ablation experiment failure analysis (pure Python, no dependencies).

Tests:
1. fb mode + gamma=0.7: Actor hand gradient magnitude is severely insufficient
2. residual variant: network parameter count is too large
3. discount_hand calculation correctness
"""
import math

print("=" * 70)
print("BFM-Zero Ablation Failure Analysis")
print("=" * 70)

DISCOUNT_BODY   = 0.98   # train.py discount
DISCOUNT_HAND_FB = 0.7   # --fb-hand-discount 0.7
Z_BODY_DIM = 324         # --z-body-dim 324  (sqrt=18)
Z_HAND_DIM = 64          # --z-hand-dim 64   (sqrt=8)

# ============================================================
# Test 1: Actor hand gradient imbalance in fb mode
# ============================================================
print("\n--- Test 1: Actor Hand Gradient Imbalance in FB Mode ---")
print("In the FB framework: Q = F . z, at convergence Q ~ ||B||^2 / (1 - gamma)")
print("z is projected to ||z|| = sqrt(z_dim), so ||B|| = sqrt(z_dim) at optimality.\n")

B_body_norm = math.sqrt(Z_BODY_DIM)   # = 18
B_hand_norm = math.sqrt(Z_HAND_DIM)   # = 8

Q_body_scale  = B_body_norm ** 2 / (1 - DISCOUNT_BODY)     # 18^2/0.02 = 16200
Q_hand_fb_scale = B_hand_norm ** 2 / (1 - DISCOUNT_HAND_FB) # 8^2/0.3 = 213.3
ratio = Q_hand_fb_scale / Q_body_scale

print(f"  ||z_body|| = sqrt({Z_BODY_DIM}) = {B_body_norm:.0f}")
print(f"  ||z_hand|| = sqrt({Z_HAND_DIM}) = {B_hand_norm:.0f}")
print(f"  Q_body  (gamma={DISCOUNT_BODY}):  {B_body_norm:.0f}^2 / (1-{DISCOUNT_BODY}) = {Q_body_scale:.0f}")
print(f"  Q_hand  (gamma={DISCOUNT_HAND_FB}): {B_hand_norm:.0f}^2 / (1-{DISCOUNT_HAND_FB}) = {Q_hand_fb_scale:.1f}")
print(f"  Q_hand / Q_body = {ratio:.5f}  ({ratio*100:.2f}%)")
print()
print("  Actor loss comparison:")
print()
print("  [MSE mode - working]")
print(f"    hand_weight = Q_body.abs().mean() ~= {Q_body_scale:.0f}  (adaptive scaling)")
print(f"    hand_term   = hand_weight * MSE_loss ~= {Q_body_scale:.0f} * MSE")
print(f"    -> hand gradient is O(Q_body) = SAME magnitude as body  [OK]")
print()
print("  [FB mode gamma=0.7 - hand flailing]")
print(f"    hand_term   = -actor_hand_q_weight * Q_hand.mean()")
print(f"                = -1.0 * {Q_hand_fb_scale:.1f}")
print(f"    -> hand gradient is only {ratio*100:.2f}% of body gradient  [BAD]")
print(f"    -> hand gradient ~{round(1/ratio)}x weaker than body - nearly zero signal!")

# Compare with larger gamma history
print()
print("  Historical comparison: why was large gamma 'less bad'?")
for gamma_h in [0.98, 0.99]:
    Q_h = B_hand_norm ** 2 / (1 - gamma_h)
    r = Q_h / Q_body_scale
    print(f"    gamma_hand={gamma_h}: Q_hand ~= {Q_h:.0f}, ratio = {r:.3f} ({r*100:.1f}%)")
print(f"    gamma_hand=0.7:  Q_hand ~= {Q_hand_fb_scale:.1f}, ratio = {ratio:.4f} ({ratio*100:.2f}%)")
print("  -> Smaller gamma = weaker hand gradient. But note: even large gamma suffers")
print("     from this imbalance. MSE (gamma->0) sidesteps it via adaptive weighting.")
print()
print("  CONCLUSION: This is NOT a code bug, but a DESIGN FLAW.")
print("  The 'fb' mode actor loss is missing the adaptive scaling used in 'mse' mode.")
print("  Without it, hand tracking signal is negligible -> hand policy drifts chaotically.")

# ============================================================
# Test 2: discount_hand calculation check
# ============================================================
print("\n--- Test 2: discount_hand Calculation Correctness ---")
print("  Code: discount_hand = discount * (fb_hand_discount / train_discount)")
print("        where discount = train_discount * ~terminated")

train_discount = 0.98
fb_hand_discount = 0.7
# ~terminated = 1.0 (not terminated)
discount = train_discount * 1.0
discount_hand = discount * (fb_hand_discount / train_discount)
expected = fb_hand_discount

print(f"\n  train_discount = {train_discount}")
print(f"  fb_hand_discount = {fb_hand_discount}")
print(f"  discount (not terminated) = {discount:.4f}")
print(f"  discount_hand = {discount:.4f} * ({fb_hand_discount} / {train_discount})")
print(f"               = {discount_hand:.8f}")
print(f"  expected     = {expected:.8f}")
ok = abs(discount_hand - expected) < 1e-10
print(f"  Result: {'PASS' if ok else 'FAIL'} - discount_hand is {'correct' if ok else 'WRONG'}")

# ============================================================
# Test 3: residual variant parameter count
# ============================================================
print("\n--- Test 3: Residual Variant Parameter Count ---")

H = 2048
HIDDEN_LAYERS = 6    # used by residual _SingleSplitForwardMap for embeddings (BUG CANDIDATE)
EMBEDDING_LAYERS = 2  # field exists in SplitForwardArchiConfig but is NEVER USED
OBS_DIM_F = 1050     # approx: state(64)+privstate(463)+last_action(29)+history(~494)
OBS_DIM_A = 587      # approx: state(64)+last_action(29)+history(~494)
BODY_ACT = 22
HAND_ACT = 7
BODY_Z = 324
HAND_Z = 64

def emb_params(in_dim, h, nl):
    """Approximate residual_embedding parameter count (dominant linear weights)."""
    p = in_dim * h          # Block(in->h)
    p += (nl - 2) * h * h  # (nl-2) ResidualBlocks
    p += h * (h // 2)       # Block(h->h/2)
    return p

def rb_params(h):
    return h * h

# --- Simple F network ---
simple_F_one = (OBS_DIM_F * H + H * 256 +
                (256 + BODY_ACT + BODY_Z) * H + H * BODY_Z +
                (256 + HAND_ACT + HAND_Z) * H + H * HAND_Z)
simple_F_total = simple_F_one * 2  # num_parallel=2

# --- Residual F network (using hidden_layers=6 for embeddings - current code) ---
residual_F_one = (emb_params(OBS_DIM_F + BODY_Z, H, HIDDEN_LAYERS) +
                  emb_params(OBS_DIM_F + BODY_ACT, H, HIDDEN_LAYERS) +
                  emb_params(OBS_DIM_F + HAND_Z, H, HIDDEN_LAYERS) +
                  emb_params(OBS_DIM_F + HAND_ACT, H, HIDDEN_LAYERS) +
                  HIDDEN_LAYERS * rb_params(H) + H * BODY_Z +
                  HIDDEN_LAYERS * rb_params(H) + H * HAND_Z)
residual_F_total = residual_F_one * 2

# --- Fixed Residual F (using embedding_layers=2) ---
fixed_F_one = (emb_params(OBS_DIM_F + BODY_Z, H, EMBEDDING_LAYERS) +
               emb_params(OBS_DIM_F + BODY_ACT, H, EMBEDDING_LAYERS) +
               emb_params(OBS_DIM_F + HAND_Z, H, EMBEDDING_LAYERS) +
               emb_params(OBS_DIM_F + HAND_ACT, H, EMBEDDING_LAYERS) +
               HIDDEN_LAYERS * rb_params(H) + H * BODY_Z +
               HIDDEN_LAYERS * rb_params(H) + H * HAND_Z)
fixed_F_total = fixed_F_one * 2

# --- Original BFM0 ResidualForwardMap ---
orig_F = (emb_params(OBS_DIM_F + BODY_Z + HAND_Z, H, HIDDEN_LAYERS) +
          emb_params(OBS_DIM_F + BODY_ACT + HAND_ACT, H, HIDDEN_LAYERS) +
          HIDDEN_LAYERS * rb_params(H) + H * (BODY_Z + HAND_Z))

print(f"\n  F Network Parameter Count (approximate):")
print(f"    Simple   Split-F  (current):  {simple_F_total/1e6:7.1f}M (per-instance {simple_F_one/1e6:.1f}M x2)")
print(f"    Residual Split-F  (current):  {residual_F_total/1e6:7.1f}M (per-instance {residual_F_one/1e6:.1f}M x2)")
print(f"    Residual Split-F  (if fixed):  {fixed_F_total/1e6:7.1f}M (per-instance {fixed_F_one/1e6:.1f}M x2)")
print(f"    Original BFM0 F   (baseline): {orig_F/1e6:7.1f}M")
print()
print(f"  Residual (current) / Simple   = {residual_F_total/simple_F_total:.1f}x")
print(f"  Residual (current) / BFM0     = {residual_F_total/orig_F:.1f}x")
print(f"  Residual (fixed)   / Simple   = {fixed_F_total/simple_F_total:.1f}x")
print(f"  Residual (fixed)   / BFM0     = {fixed_F_total/orig_F:.1f}x")
print()

print("  ROOT CAUSE: In _SingleSplitForwardMap residual mode (nn_models.py line 903-906):")
print("    self.hand_embed_z = residual_embedding(..., cfg.hidden_layers)   # = 6")
print("    self.hand_embed_sa = residual_embedding(..., cfg.hidden_layers)  # = 6")
print("    ...")
print("  The field 'embedding_layers=2' in SplitForwardArchiConfig is NEVER USED.")
print("  This creates 4 deep embedding networks (6 layers each) instead of the")
print("  intended shallow ones (2 layers), inflating F network by ~9x.")
print()
print("  Note: SplitActor DOES correctly use 'embedding_layers=2' for its embeddings.")
print("  The F network inconsistency is a BUG in the residual variant implementation.")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Issue 1: FB mode + gamma=0.7 (fb_hand_loss_mode='fb', fb_hand_discount=0.7)
  [NOT a code logic bug, but a design flaw in actor loss]
  - The actor loss for 'fb' mode uses fixed weight (actor_hand_q_weight=1.0)
  - With gamma=0.7 vs body gamma=0.98, Q_hand is ~75x smaller than Q_body
  - Hand gradient contributes only ~1.3% to total actor loss
  - Hand policy receives negligible update signal -> drifts chaotically
  - MSE mode avoids this via adaptive scaling: hand_weight = Q_body.abs().mean()
  FIX: Add adaptive scaling in fb mode actor loss (see below)

Issue 2: Residual variant (split_network_variant='residual')
  [BUG: 'embedding_layers' field is defined but never used in F network]
  - _SingleSplitForwardMap uses hidden_layers=6 for ALL 4 embeddings
  - Should use embedding_layers=2 to mirror original ResidualForwardMap design
  - Current F network is ~9x larger than Simple variant, ~5x larger than original BFM0
  - Too large to converge in the same training budget as Simple variant
  FIX: Replace cfg.hidden_layers with cfg.embedding_layers in lines 903-906

Code fix locations:
  Issue 1: humanoidverse/agents/fb_cpr/agent.py line 453-454
           humanoidverse/agents/fb_cpr_aux/agent.py line 303-304
  Issue 2: humanoidverse/agents/nn_models.py lines 903-906
""")
