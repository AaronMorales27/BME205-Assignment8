#!/usr/bin/env python3
"""
Wright-Fisher forward simulation with selection (Part 1 of assignment).

Usage:
    python aaron_morales_assignment7_part1.py wright_fisher_config.csv wright_fisher_demography.csv --seed 12345 --output forward_results.tsv

Notes / assumptions:
- Haploid population (each individual carries a single allele at each site).
- Infinite-sites-like tracking: we track segregating sites only (not full genomes).
- Watterson's theta uses sample size n = current population size N (as described in explanation).
"""

import argparse
import pandas as pd
import numpy as np

import csv
import math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Wright-Fisher forward simulator with selection")
    p.add_argument("config", help="Config CSV (parameter,value)")
    p.add_argument("demography", help="Demography CSV (generation,population_size)")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--output", required=True, help="Output TSV for forward results")
    return p.parse_args()

def read_config(config_path):
    df = pd.read_csv(config_path, dtype=str)
    cfg = dict(zip(df['parameter'], df['value']))
    # convert numbers to appropriate types
    mu = float(cfg['mutation_rate'])
    L = int(cfg['sequence_length'])
    total_generations = int(cfg['total_generations'])
    s = float(cfg['selection_coefficient'])
    beneficial_time = int(cfg['beneficial_mutation_time'])
    return mu, L, total_generations, s, beneficial_time

def read_demography(demo_path, total_generations):
    d = pd.read_csv(demo_path)
    # expects columns: generation,population_size
    # build per-generation population sizes
    pop_sizes = np.empty(total_generations+1, dtype=int)  # include generation 0..total_generations
    # ensure sorted by generation
    d = d.sort_values(by=d.columns[0]).reset_index(drop=True)
    gen_col = d.columns[0]
    pop_col = d.columns[1]
    # iterate through intervals
    for idx in range(len(d)):
        g = int(d.loc[idx, gen_col])
        N = int(d.loc[idx, pop_col])
        # next change generation:
        next_g = total_generations + 1
        if idx + 1 < len(d):
            next_g = int(d.loc[idx+1, gen_col])
        # apply N for generations g .. next_g-1
        end = min(next_g, total_generations+1)
        pop_sizes[g:end] = N
    return pop_sizes

def harmonic_number(n):
    # a_n = sum_{i=1}^{n-1} 1/i
    if n < 2:
        return 0.0
    # use approximation for very large n? direct sum is fine for n up to 10000
    return sum(1.0/i for i in range(1, n))

def write_output_rows(output_path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6f")

def main():
    args = parse_args()
    np.random.seed(args.seed)

    mu, L, total_generations, s, beneficial_time = read_config(args.config)
    pop_sizes = read_demography(args.demography, total_generations)

    # State: dictionary of segregating mutations
    # Each entry -> {position:int, count:int, beneficial:bool}
    mutations = {}
    next_mut_id = 0
    beneficial_mut_id = None

    rows = []
    # Pre-record generation 0 (no mutations at start)
    # We'll still compute and print stats for generation 0
    for gen in range(total_generations + 1):
        N = int(pop_sizes[gen])
        # 1) Introduce new neutral mutations (Poisson with mean mu * L * N)
        mean_new = mu * L * N
        num_new = np.random.poisson(mean_new) if mean_new > 0 else 0
        for _ in range(num_new):
            mutations[next_mut_id] = {
                "position": np.random.randint(0, L),
                "count": 1,
                "beneficial": False
            }
            next_mut_id += 1

        # 2) Introduce beneficial mutation at configured generation
        if gen == beneficial_time:
            # create a beneficial mutation that starts in single individual
            mutations[next_mut_id] = {
                "position": np.random.randint(0, L),
                "count": 1,
                "beneficial": True
            }
            beneficial_mut_id = next_mut_id
            next_mut_id += 1

        # 3) Compute selection-adjusted frequencies and sample next generation counts
        # We'll create a new dict with updated counts after drift
        updated = {}
        for mid, rec in mutations.items():
            # current frequency
            # avoid division by zero: if N==0 skip (but assignment N never zero)
            p = rec["count"] / N if N > 0 else 0.0

            # selection adjustment only for beneficial mutation
            if rec.get("beneficial", False):
                # haploid selection update:
                # p_sel = p*(1+s) / (p*(1+s) + (1-p)*1) = p*(1+s) / (1 + p*s)
                denom = 1.0 + p * s
                p_eff = (p * (1.0 + s)) / denom if denom != 0 else p
            else:
                p_eff = p  # neutral

            # sample count in next generation via Binomial(N, p_eff)
            # for numerical stability clamp p_eff into [0,1]
            p_eff = max(0.0, min(1.0, p_eff))
            new_count = np.random.binomial(N, p_eff) if N > 0 else 0

            # keep only polymorphic sites (0 < count < N)
            if 0 < new_count < N:
                updated[mid] = {"position": rec["position"], "count": int(new_count), "beneficial": rec.get("beneficial", False)}
            # if new_count == N, site is fixed derived; we drop it from segregating list
            # if new_count == 0, site lost; drop it as well

        mutations = updated

        # 4) Compute diversity statistics
        S = len(mutations)  # segregating sites
        a_n = harmonic_number(N)  # use current population size as sample size
        theta_watterson = (S / a_n) if (a_n > 0 and S > 0) else 0.0

        # nucleotide diversity pi per site = (1/L) * sum_sites 2 p (1-p)
        pi_num = 0.0
        beneficial_freq = 0.0
        for rec in mutations.values():
            p = rec["count"] / N if N > 0 else 0.0
            pi_num += 2.0 * p * (1.0 - p)
            if rec.get("beneficial", False):
                beneficial_freq = p

        nucleotide_diversity = pi_num / L

        # num_mutations in output: report total currently segregating
        num_mutations = S

        # Save row
        row = {
            "generation": gen,
            "population_size": N,
            "num_mutations": num_mutations,
            "nucleotide_diversity": nucleotide_diversity,
            "theta_watterson": theta_watterson,
            "beneficial_freq": beneficial_freq
        }
        rows.append(row)

    # Write output TSV
    write_output_rows(args.output, rows)
    # print(f"Wright-Fisher forward simulation complete. Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
