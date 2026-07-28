"""Theorem 5: minimax optimality of the simple bound in regime R2.

Pins the constructive proof: two populations indistinguishable from
single-draw binary data whose clean means differ by exactly pi*Lambda,
and the certificate ceiling Gamma* = Delta/(pi*(1+rho)).
"""
import numpy as np

from contamsens import gamma_star, max_bias_simple

PI, LAM, MU = 0.1, 0.3, 0.55


def test_minimax_indistinguishable_pair():
    # P1: clean, p* = mu everywhere. P2: pi mass contaminated at p* = 0,
    # delta = Lambda (observed p = Lambda); clean mass at (mu - pi*Lam)/(1-pi).
    p_clean_star = (MU - PI * LAM) / (1 - PI)
    assert 0.0 <= p_clean_star <= 1.0
    mu2 = PI * LAM + (1 - PI) * p_clean_star          # observable mean of P2
    assert abs(mu2 - MU) < 1e-12                      # indistinguishable
    theta1, theta2 = MU, (1 - PI) * p_clean_star      # clean means
    gap = theta1 - theta2
    assert abs(gap - PI * LAM) < 1e-12                # width attained
    assert abs(max_bias_simple(LAM, PI) - gap) < 1e-12  # bound = sharp width


def test_minimax_no_narrower_certificate():
    # A margin Delta flips at any Lambda > Delta/(pi*(1+rho)) and at none
    # below: the simple Gamma* equals the information-theoretic ceiling.
    rng = np.random.default_rng(0)
    n, delta, rho = 4000, 0.02, 0.2
    ya = (rng.random(n) < 0.60).astype(float)
    yb = ya.copy()
    flip = rng.choice(n, size=int(round(delta * n)), replace=False)
    yb[flip] = 0.0                                    # margin ~ delta
    d = ya.mean() - yb.mean()
    g = gamma_star(ya, yb, PI, simple=True, rho=rho)
    assert abs(g - d / (PI * (1 + rho))) < 1e-9
    # ceiling: worst-case bias at Lambda just above g overturns the margin
    lam_hi = g * 1.01
    assert PI * lam_hi * (1 + rho) > d
    assert PI * (g * 0.99) * (1 + rho) < d
