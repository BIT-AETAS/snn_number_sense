#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Lagrange Information Theory Analysis')
parser.add_argument('--min_n', type=int, default=1, help='Minimum numerosity')
parser.add_argument('--max_n', type=int, default=30, help='Maximum numerosity')
parser.add_argument('--min_k', type=int, default=1, help='Minimum k value')
parser.add_argument('--max_k', type=int, default=31, help='Maximum k value')
parser.add_argument('--T', type=int, default=4, help='Number of time steps')
parser.add_argument('--info_bound', type=float, default=2.0, help='Information bound for KL divergence')
parser.add_argument('--n_steps', type=int, default=1500, help='Number of steps for gradient descent')
args = parser.parse_args()

def KL(p,q):
    # KL divergence with prior p and posterior q
    return np.sum(q * (np.log2(q) - np.log2(p)), axis=1)

def compute_q_nk(ns, ks, p_n, p_k, T, lam, sm=1e-10):
    # computing the posterior Q for a given lambda
    lam = lam.reshape((len(lam),1))
    p_n = p_n.reshape((len(p_n),1))
    q_nk = -((p_n * (ns-ks)**2.)/(lam * T))
    q_nk = p_k * np.exp(q_nk)
    q_nk = q_nk / np.sum(q_nk, axis=1).reshape(len(q_nk), 1)
    q_nk += sm
    q_nk = q_nk / np.sum(q_nk, axis=1).reshape(len(q_nk), 1)

    return q_nk

def find_q_nk(ns_arr, ks, p_n, p_k, T, info_bound, n_steps=1500):
    # uses gradient descent to find lambdas that
    # get KL(Q||P) as close to the bound "info_bound" as possible

    lams = np.ones_like(ns_arr) * 0.5
    q_nk = compute_q_nk(ns_arr, ks, p_n, p_k, T, lams)
    ents = KL(p_k, q_nk)

    for _ in range(n_steps):
        diffs = ents - info_bound
        deltas = diffs * 0.025
        lams = np.exp(np.log(lams) + deltas.reshape(len(deltas), 1))
        q_nk = compute_q_nk(ns_arr, ks, p_n, p_k, T, lams)
        ents = KL(p_k, q_nk)

    return q_nk

def P(x, a):
    p = 1. / (x**a)
    return p / np.sum(p)

if __name__ == "__main__":
    ks = np.arange(args.min_k, args.max_k, 1)
    ns = np.arange(args.min_n, args.max_n + 1, 1)
    ns = ns.reshape((len(ns), 1))

    p_ks = P(ks, 1.0)
    p_ns = P(ns, 1.0)

    Q = find_q_nk(ns, ks, p_ns, p_ks, args.T, args.info_bound, n_steps=args.n_steps)
    print(Q)
    
