"""
Cross-task analysis: build the per-unit overlap tensor and derive
universality scores, a task similarity matrix and a task hierarchy.

Input: dict task_name -> {unit_id -> svd result dict} (from weighted_unit_svd).
Output of run_analysis(): dict with the overlap/significance tensor,
universality scores, ranked units, task similarity, linkage and distance.
No plotting here. Running the module directly executes a synthetic self-test:
    python cross_task/cross_task_analysis.py
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from subspace_overlap import (
    subspace_overlap,
    BaselineCache,
)


# Step 1: build the overlap tensor

def build_overlap_tensor(
    task_subspaces: Dict[str, Dict[Any, Dict[str, Any]]],
    baseline_cache: Optional[BaselineCache] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute the (n_units, n_tasks, n_tasks) overlap tensor and the
    corresponding significance tensor.

    Args:
        task_subspaces: dict[task_name -> dict[unit_id -> svd result]]
                        as produced by weighted_unit_svd.
                        All tasks must contain the same set of unit_ids.
        baseline_cache: optional precomputed BaselineCache. If None, a
                        fresh one is created.
        verbose:        print progress.

    Returns:
        dict with:
            'overlap':       (n_units, n_tasks, n_tasks) float32
            'baseline':      (n_units, n_tasks, n_tasks) float32
            'significant':   (n_units, n_tasks, n_tasks) bool
            'unit_ids':      list of unit identifiers (length n_units)
            'unit_meta':     list of dicts with type/layer/d_k per unit
            'task_names':    list of task names (length n_tasks)
            'rho_matrix':    (n_units, n_tasks) int, rank per (unit, task)
    """
    task_names = sorted(task_subspaces.keys())
    if not task_names:
        raise ValueError("task_subspaces is empty")

    # Use the first task to define unit ordering. Verify all tasks share it.
    unit_ids = list(task_subspaces[task_names[0]].keys())
    for t in task_names[1:]:
        missing = set(unit_ids) - set(task_subspaces[t].keys())
        extra   = set(task_subspaces[t].keys()) - set(unit_ids)
        if missing or extra:
            raise ValueError(
                f"Task '{t}' has mismatched units: "
                f"missing {len(missing)}, extra {len(extra)}"
            )

    n_units = len(unit_ids)
    n_tasks = len(task_names)

    if baseline_cache is None:
        baseline_cache = BaselineCache(n_samples=200, quantile=0.99,
                                       seed=0, verbose=False)

    overlap  = np.zeros((n_units, n_tasks, n_tasks), dtype=np.float32)
    baseline = np.zeros((n_units, n_tasks, n_tasks), dtype=np.float32)
    rho_mat  = np.zeros((n_units, n_tasks), dtype=np.int32)
    importance = np.zeros((n_units, n_tasks), dtype=np.float32)

    unit_meta: List[dict] = []

    if verbose:
        print(f"build_overlap_tensor: {n_units} units x "
              f"{n_tasks} tasks x {n_tasks} tasks")

    for u_idx, uid in enumerate(unit_ids):
        # Cache rho and importance per (unit, task)
        for t_idx, t in enumerate(task_names):
            rho_mat[u_idx, t_idx] = task_subspaces[t][uid]['rho']
            # 'importance' is only present in weighted_unit_svd output;
            # default to 1.0 (uniform) when absent.
            importance[u_idx, t_idx] = task_subspaces[t][uid].get(
                'importance', 1.0)

        # All tasks share the same d_k for this unit
        d_k = task_subspaces[task_names[0]][uid]['d_k']
        meta = dict(task_subspaces[task_names[0]][uid].get('meta', {}))
        meta['d_k'] = d_k
        unit_meta.append(meta)

        # Pairwise overlaps for this unit
        for i in range(n_tasks):
            U_i = task_subspaces[task_names[i]][uid]['U']
            for j in range(n_tasks):
                if j < i:
                    overlap[u_idx, i, j]  = overlap[u_idx, j, i]
                    baseline[u_idx, i, j] = baseline[u_idx, j, i]
                    continue
                if i == j:
                    # Self-overlap is 1 if rho > 0, else 0 (unused downstream).
                    overlap[u_idx, i, j] = 1.0 if rho_mat[u_idx, i] > 0 else 0.0
                    baseline[u_idx, i, j] = 0.0
                    continue
                U_j = task_subspaces[task_names[j]][uid]['U']
                overlap[u_idx, i, j] = subspace_overlap(U_i, U_j)
                baseline[u_idx, i, j] = baseline_cache.get(
                    d=d_k,
                    p=int(rho_mat[u_idx, i]),
                    q=int(rho_mat[u_idx, j]),
                )

        if verbose and (u_idx + 1) % 25 == 0:
            print(f"  [{u_idx+1:3d}/{n_units}] last: {uid}")

    significant = overlap > baseline
    # Don't count diagonal as significant
    diag = np.eye(n_tasks, dtype=bool)[None, :, :]
    significant = significant & ~diag

    if verbose:
        n_baselines = len(baseline_cache)
        print(f"  done. {n_baselines} unique baseline shapes computed.")

    return {
        'overlap':     overlap,
        'baseline':    baseline,
        'significant': significant,
        'unit_ids':    unit_ids,
        'unit_meta':   unit_meta,
        'task_names':  task_names,
        'rho_matrix':  rho_mat,
        'importance':  importance,    # (n_units, n_tasks); 1.0 if the input
                                       # has no importance field
    }


# Step 2: per-unit universality scores

def universality_scores(tensor: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Per-unit universality:
        - min:  the minimum overlap across all off-diagonal task pairs
                (a unit only counts as universal if ALL pairs are high)
        - mean: the mean overlap across off-diagonal pairs
        - n_significant: count of off-diagonal pairs above baseline
    """
    overlap     = tensor['overlap']            # (U, T, T)
    significant = tensor['significant']         # (U, T, T)
    n_units, n_tasks, _ = overlap.shape

    # Mask off the diagonal
    off_diag = ~np.eye(n_tasks, dtype=bool)
    off_diag_b = off_diag[None, :, :].repeat(n_units, axis=0)

    off_vals = overlap[off_diag_b].reshape(n_units, -1)
    n_pairs = off_vals.shape[1]                # = T*(T-1), counts (i,j) twice
    # min/mean over the symmetric tensor is the same as over unique pairs.

    return {
        'min':           off_vals.min(axis=1),
        'mean':          off_vals.mean(axis=1),
        'max':           off_vals.max(axis=1),
        'n_significant': significant.reshape(n_units, -1).sum(axis=1) // 2,
        'n_pairs_total': n_pairs // 2,
    }


def rank_universal_units(
    tensor: Dict[str, Any],
    by: str = 'min',
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return units sorted by universality, descending.

    Args:
        tensor: output of build_overlap_tensor.
        by:     'min' (most stringent) or 'mean' or 'n_significant'.
        top_k:  return only the top k units (None = all).
    """
    scores = universality_scores(tensor)
    if by not in scores:
        raise ValueError(f"by must be one of {list(scores)}, got {by!r}")
    score_arr = scores[by]
    order = np.argsort(-score_arr)             # descending
    if top_k is not None:
        order = order[:top_k]

    out = []
    for rank, u_idx in enumerate(order):
        out.append({
            'rank':           rank,
            'unit_id':        tensor['unit_ids'][u_idx],
            'unit_meta':      tensor['unit_meta'][u_idx],
            'score_min':      float(scores['min'][u_idx]),
            'score_mean':     float(scores['mean'][u_idx]),
            'score_max':      float(scores['max'][u_idx]),
            'n_significant':  int(scores['n_significant'][u_idx]),
            'n_pairs_total':  int(scores['n_pairs_total']),
            'rho_per_task':   tensor['rho_matrix'][u_idx].tolist(),
            'importance_per_task': tensor['importance'][u_idx].tolist(),
            'importance_min': float(tensor['importance'][u_idx].min()),
            'importance_geomean': float(
                np.exp(np.log(np.maximum(tensor['importance'][u_idx], 1e-12)).mean())
            ),
            'overlap_matrix': tensor['overlap'][u_idx].tolist(),
        })
    return out


# Step 3: task-level similarity matrix

def task_similarity_matrix(
    tensor: Dict[str, Any],
    mode: str = 'significant_mean',
) -> np.ndarray:
    """
    Aggregate per-unit overlaps into a (n_tasks, n_tasks) similarity matrix.

    Modes:
        'mean':              simple mean over all units
        'significant_mean':  mean only over units where overlap > baseline
                             (ignores the noise floor)
        'significant_count': fraction of units where overlap is significant
        'top_k_mean':        mean over the top-k units (k = n_units // 4)
        'importance_weighted':
            weighted mean using joint per-task importance as weights:
                w_u^(s,t) = sqrt(I_u^s * I_u^t)
                sim(s,t) = sum_u(w_u^(s,t) * overlap_u^(s,t)) / sum_u(w_u^(s,t))
            Requires 'importance' fields in task_subspaces. The geometric
            mean means a unit only contributes when it is important in
            both tasks.

    Returns:
        (n_tasks, n_tasks) float32 matrix. Diagonal is 1.
    """
    overlap     = tensor['overlap']
    significant = tensor['significant']
    importance  = tensor['importance']         # (n_units, n_tasks)
    n_units, n_tasks, _ = overlap.shape

    if mode == 'mean':
        S = overlap.mean(axis=0)

    elif mode == 'significant_mean':
        weights = significant.astype(np.float32)
        weighted_sum = (overlap * weights).sum(axis=0)
        denom = weights.sum(axis=0)
        denom = np.where(denom == 0, 1.0, denom)
        S = weighted_sum / denom

    elif mode == 'significant_count':
        S = significant.mean(axis=0).astype(np.float32)

    elif mode == 'top_k_mean':
        k = max(1, n_units // 4)
        S = np.zeros((n_tasks, n_tasks), dtype=np.float32)
        for i in range(n_tasks):
            for j in range(n_tasks):
                vals = overlap[:, i, j]
                top = np.sort(vals)[-k:]
                S[i, j] = top.mean()

    elif mode == 'importance_weighted':
        # Check that importance is non-trivial (not all 1.0 from defaults)
        if np.allclose(importance, 1.0):
            print("WARNING: importance_weighted mode requested but all "
                  "importance values are 1.0 (task subspaces probably lack "
                  "importance fields). "
                  "Falling back to uniform weighting (= 'mean' mode).")
        S = np.zeros((n_tasks, n_tasks), dtype=np.float32)
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    S[i, j] = 1.0
                    continue
                # Joint weight: geometric mean of per-task importance
                w_ij = np.sqrt(importance[:, i] * importance[:, j])  # (n_units,)
                num = (w_ij * overlap[:, i, j]).sum()
                den = w_ij.sum()
                S[i, j] = num / den if den > 0 else 0.0

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # Force diagonal to 1, force symmetry against numerical drift
    np.fill_diagonal(S, 1.0)
    S = 0.5 * (S + S.T)
    return S.astype(np.float32)


# Step 4: hierarchical clustering of tasks

def hierarchical_clustering_tasks(
    similarity: np.ndarray,
    method: str = 'average',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert task similarity matrix to a distance matrix and run scipy
    hierarchical clustering.

    Args:
        similarity: (n_tasks, n_tasks) symmetric, in [0, 1].
        method:     scipy linkage method (default 'average').

    Returns:
        (linkage_matrix, distance_matrix)
        linkage_matrix is the input to scipy.cluster.hierarchy.dendrogram.
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage

    n = similarity.shape[0]
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    # Ensure exact symmetry for squareform
    distance = 0.5 * (distance + distance.T)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method=method)
    return Z, distance


# Top-level driver

def run_analysis(
    task_subspaces: Dict[str, Dict[Any, Dict[str, Any]]],
    baseline_n_samples: int = 200,
    baseline_quantile: float = 0.99,
    similarity_mode: str = 'significant_mean',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full cross-task analysis end to end.

    Args:
        task_subspaces:    dict[task -> dict[unit_id -> svd result]]
        baseline_n_samples: Monte Carlo samples for random baseline
        baseline_quantile:  significance threshold quantile
        similarity_mode:    how to aggregate per-unit overlaps into the
                            task similarity matrix (see task_similarity_matrix)

    Returns:
        AnalysisResult dict with:
            'tensor':              overlap tensor dict (build_overlap_tensor output)
            'universality':        universality scores dict
            'ranked_units':        list of unit dicts sorted by universality
            'task_similarity':     (n_tasks, n_tasks) matrix
            'task_linkage':        scipy linkage matrix
            'task_distance':       (n_tasks, n_tasks) distance matrix
            'task_names':          list of task names
    """
    cache = BaselineCache(
        n_samples=baseline_n_samples,
        quantile=baseline_quantile,
        seed=0,
        verbose=verbose,
    )

    if verbose:
        print("=" * 70)
        print("STEP 1: building overlap tensor")
        print("=" * 70)
    tensor = build_overlap_tensor(task_subspaces, cache, verbose=verbose)

    if verbose:
        print()
        print("=" * 70)
        print("STEP 2: per-unit universality scores")
        print("=" * 70)
    univ = universality_scores(tensor)
    ranked = rank_universal_units(tensor, by='min', top_k=20)

    if verbose:
        n_units = len(tensor['unit_ids'])
        n_universal = int((univ['n_significant'] == univ['n_pairs_total']).sum())
        print(f"  units where ALL task pairs are significant: "
              f"{n_universal}/{n_units}")
        print(f"  top 5 by min-universality:")
        for r in ranked[:5]:
            print(f"    {r['unit_id']}  min={r['score_min']:.3f}  "
                  f"mean={r['score_mean']:.3f}  "
                  f"sig={r['n_significant']}/{r['n_pairs_total']}  "
                  f"meta={r['unit_meta']}")

    if verbose:
        print()
        print("=" * 70)
        print(f"STEP 3: task similarity matrix (mode={similarity_mode!r})")
        print("=" * 70)
    sim = task_similarity_matrix(tensor, mode=similarity_mode)
    if verbose:
        print(f"  task_names: {tensor['task_names']}")
        print("  similarity matrix:")
        for i, t in enumerate(tensor['task_names']):
            row = "  ".join(f"{sim[i, j]:.3f}" for j in range(sim.shape[1]))
            print(f"    {t:>20s} | {row}")

    if verbose:
        print()
        print("=" * 70)
        print("STEP 4: hierarchical clustering of tasks")
        print("=" * 70)
    Z, D = hierarchical_clustering_tasks(sim)
    if verbose:
        print("  linkage matrix (cluster_a, cluster_b, distance, n_obs):")
        for row in Z:
            print(f"    {row}")
        print()
        print("  reading the dendrogram:")
        for step, row in enumerate(Z):
            a, b, dist, _ = row
            print(f"    merge {step}: clusters {int(a)} + {int(b)} "
                  f"at distance {dist:.3f}")

    return {
        'tensor':          tensor,
        'universality':    univ,
        'ranked_units':    ranked,
        'task_similarity': sim,
        'task_linkage':    Z,
        'task_distance':   D,
        'task_names':      tensor['task_names'],
    }


# Self-test on synthetic data

if __name__ == "__main__":
    """
    Synthetic test: 4 tasks, 8 units. Planted ground truth:
        units 0,1,2: family A (tasks A,B share an identical subspace)
        units 3,4,5: family B (tasks C,D share an identical subspace)
        unit 6:      universal (all 4 tasks share an identical subspace)
        unit 7:      noise (independent random subspace per task)

    Expected:
        - unit 6 ranks highest by every universality metric
        - unit 7 ranks lowest
        - sim(A,B) and sim(C,D) significantly exceed cross-family pairs
        - dendrogram first merge is within-family (A+B or C+D)
    """
    print("=" * 70)
    print("SYNTHETIC TEST: 4 tasks, 8 units, planted hierarchy")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    d = 200
    rho = 4
    tasks = ['task_A', 'task_B', 'task_C', 'task_D']

    def random_subspace():
        Q, _ = np.linalg.qr(rng.standard_normal((d, rho)).astype(np.float32))
        return Q

    family_A = random_subspace()
    family_B = random_subspace()
    universal = random_subspace()

    task_subspaces = {t: {} for t in tasks}
    for unit_id in range(8):
        for t in tasks:
            if unit_id in (0, 1, 2):
                U = family_A if t in ('task_A', 'task_B') else random_subspace()
            elif unit_id in (3, 4, 5):
                U = family_B if t in ('task_C', 'task_D') else random_subspace()
            elif unit_id == 6:
                U = universal
            else:                                        # unit 7: noise
                U = random_subspace()

            task_subspaces[t][f'unit_{unit_id}'] = {
                'U': U, 'sigma': np.ones(rho, dtype=np.float32),
                'rho': rho, 'd_k': d, 'var_explained': 0.95,
                'meta': {'unit_id': f'unit_{unit_id}',
                         'unit_type': 'mlp_up' if unit_id % 2 == 0 else 'attn_head',
                         'layer': unit_id // 2, 'd_k': d},
            }

    # Use 'mean' aggregation over all units; 'significant_mean' would
    # equalize all task pairs through the universal unit.
    result = run_analysis(task_subspaces, baseline_n_samples=100,
                          similarity_mode='mean', verbose=True)

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check 1: ranking by mean universality
    ranked_by_mean = sorted(
        result['ranked_units'],
        key=lambda r: -r['score_mean'],
    )
    ranked_ids_mean = [r['unit_id'] for r in ranked_by_mean]
    print(f"  ranking by mean universality: {ranked_ids_mean}")
    assert ranked_ids_mean[0]  == 'unit_6', \
        f"unit_6 should be most universal, got {ranked_ids_mean[0]}"
    assert ranked_ids_mean[-1] == 'unit_7', \
        f"unit_7 should be least universal, got {ranked_ids_mean[-1]}"
    print("  OK: universal & noise units correctly ranked")

    # Check 2: within-family > cross-family
    sim = result['task_similarity']
    tn  = result['task_names']
    iA, iB, iC, iD = tn.index('task_A'), tn.index('task_B'), \
                     tn.index('task_C'), tn.index('task_D')
    within = [sim[iA, iB], sim[iC, iD]]
    cross  = [sim[iA, iC], sim[iA, iD], sim[iB, iC], sim[iB, iD]]
    print(f"  within-family pairs: {[f'{x:.3f}' for x in within]}")
    print(f"  cross-family pairs:  {[f'{x:.3f}' for x in cross]}")
    assert min(within) > max(cross), \
        f"within-family ({min(within):.3f}) should exceed " \
        f"cross-family ({max(cross):.3f})"
    print("  OK: within-family > cross-family by every comparison")

    # Check 3: dendrogram first merge is within-family
    Z = result['task_linkage']
    first_merge = {int(Z[0, 0]), int(Z[0, 1])}
    expected_within_family = [{iA, iB}, {iC, iD}]
    print(f"  first merge cluster IDs: {first_merge}")
    print(f"  acceptable within-family pairs: {expected_within_family}")
    assert first_merge in expected_within_family, \
        "first merge should be within a family"
    print("  OK: dendrogram first merge is within-family")

    print()
    print("All synthetic tests passed.")