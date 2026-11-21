"""Block-derived, importance-weighted unit subspaces.

For task t and unit u in parent block B (rows idx_u of the block SVD):

    M_u = U_B[idx_u, :] * diag(sigma_B / sqrt(d_B))      # (d_u, rho_B)
    Ũ_u, σ̃_u, _ = SVD(M_u)

Ũ_u is an orthonormal basis in R^{d_u} used for subspace overlap, and the
unit importance I_u = sum_i (σ̃_u)_i^2 = ||M_u||_F^2 is the unit's share of
its block's energy. Compared with an independent per-unit SVD, this carries
the block structure and block energy down to the unit level.

Inputs: block SVD results (U_B, sigma_B, unit_ids, global parameter indices
in the same order as the rows of U_B) and the unit layout (units exposing
.unit_id and .get_indices()).
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np


# Expected block_results format:
#
#   block_results: Dict[block_id -> {
#       'U':              (d_block, rho_B) float32 ndarray
#       'sigma':          (rho_B,) float32 ndarray
#       'unit_ids':       List of unit_id values in this block
#       'global_indices': (d_block,) int64 ndarray of global parameter
#                         indices, in the same order as the rows of U
#   }]


def _build_unit_to_block_index(
    block_results: Dict[Any, Dict[str, Any]],
    unit_layout: List[Any],
) -> Dict[Any, tuple]:
    """
    For each unit, find:
        - which block it belongs to
        - the LOCAL row positions in that block's U corresponding to
          this unit's parameters

    Returns:
        dict[unit_id -> (block_id, local_idx ndarray)]
    """
    unit_layout_by_id = {u.unit_id: u for u in unit_layout}
    out: Dict[Any, tuple] = {}

    for block_id, block in block_results.items():
        block_global = np.asarray(block['global_indices'], dtype=np.int64)
        # global_idx -> local row; works for any ordering of block_global.
        global_to_local = {int(g): i for i, g in enumerate(block_global)}

        for uid in block['unit_ids']:
            if uid not in unit_layout_by_id:
                raise KeyError(f"block {block_id} references unknown unit {uid!r}")
            unit = unit_layout_by_id[uid]
            unit_global = np.asarray(unit.get_indices(), dtype=np.int64).ravel()

            try:
                local = np.array([global_to_local[int(g)] for g in unit_global],
                                 dtype=np.int64)
            except KeyError as e:
                raise ValueError(
                    f"unit {uid!r} (in block {block_id}) has parameter "
                    f"index {e.args[0]} that is not in the block's "
                    f"global_indices. block_finder / block_results are "
                    f"out of sync with unit_layout."
                )

            if uid in out:
                raise ValueError(
                    f"unit {uid!r} appears in multiple blocks: "
                    f"{out[uid][0]} and {block_id}. Each unit must belong "
                    f"to exactly one block."
                )
            out[uid] = (block_id, local)

    return out


# Per-unit weighted SVD

def weighted_unit_subspace(
    block_U: np.ndarray,            # (d_block, rho_B)
    block_sigma: np.ndarray,        # (rho_B,)
    local_idx: np.ndarray,          # (d_u,) int, rows of block_U for this unit
    eta_inner: float = 0.99,
    eps: float = 1e-12,
) -> Optional[Dict[str, Any]]:
    """
    Compute the contextualized subspace for one unit from its parent
    block's SVD result.

    Returns None if the unit slice is degenerate (all-zero or empty).
    Otherwise returns a dict in the same schema as unit_svd's output,
    plus an 'importance' field.
    """
    d_u = local_idx.shape[0]
    if d_u == 0 or block_U.shape[1] == 0:
        return None

    # Slice and weight: M[i, j] = U_B[local_idx[i], j] * sigma_B[j] / sqrt(d_block)
    d_block = block_U.shape[0]
    sigma_normalized = block_sigma / np.sqrt(d_block)
    M = block_U[local_idx, :].astype(np.float32) * sigma_normalized[None, :].astype(np.float32)

    fro_sq = float((M ** 2).sum())
    if fro_sq < eps:
        return None

    # Small SVD on (d_u, rho_B). Both dims are typically <= 100.
    try:
        Q, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    # Trim by inner variance threshold
    s_sq = s ** 2
    total = s_sq.sum()
    if total < eps:
        return None
    cum_ratio = np.cumsum(s_sq) / total
    rho = int(np.searchsorted(cum_ratio, eta_inner) + 1)
    rho = min(rho, len(s))

    return {
        'U':            Q[:, :rho].astype(np.float32),
        'sigma':        s[:rho].astype(np.float32),
        'rho':          rho,
        'd_k':          d_u,
        'importance':   fro_sq,                  # ||M||_F^2
        'var_explained': float(cum_ratio[rho - 1]),
    }


# Whole-task driver

def compute_task_subspaces_from_blocks(
    block_results: Dict[Any, Dict[str, Any]],
    unit_layout: List[Any],
    eta_inner: float = 0.99,
    verbose: bool = True,
) -> Dict[Any, Dict[str, Any]]:
    """
    Compute weighted unit subspaces for all units in a task.

    Args:
        block_results: dict[block_id -> {U, sigma, unit_ids, global_indices}]
                       (see schema comment at top of file)
        unit_layout:   List[UnitInfo]; must match the layout used by
                       block_finder when these blocks were built
        eta_inner:     variance threshold for the inner unit-level SVD
                       (0.99; the matrix is small, so keeping more
                       directions is cheap)

    Returns:
        dict[unit_id -> {
            'U':              (d_u, rho_u) float32, orthonormal
            'sigma':          (rho_u,) float32, weighted singular values
            'rho':            int
            'd_k':            int
            'importance':     float, ||M||_F^2 (unit's energy in its block)
            'var_explained':  float
            'meta': {
                'unit_id':       ...
                'unit_type':     ...
                'layer':         ...
                'parent_block':  block_id
                'importance':    float (duplicated for convenience)
            }
        }]

    Units that don't belong to any block, or whose slice is degenerate,
    are dropped from the output.
    """
    if verbose:
        n_blocks = len(block_results)
        n_units  = len(unit_layout)
        print(f"compute_task_subspaces_from_blocks: "
              f"{n_blocks} blocks, {n_units} units in layout")

    unit_to_block = _build_unit_to_block_index(block_results, unit_layout)
    if verbose:
        n_assigned = len(unit_to_block)
        print(f"  {n_assigned}/{len(unit_layout)} units assigned to blocks")

    unit_layout_by_id = {u.unit_id: u for u in unit_layout}
    out: Dict[Any, Dict[str, Any]] = {}
    n_dropped = 0
    importances: List[float] = []

    for uid, (block_id, local_idx) in unit_to_block.items():
        block = block_results[block_id]
        result = weighted_unit_subspace(
            block_U=np.asarray(block['U'], dtype=np.float32),
            block_sigma=np.asarray(block['sigma'], dtype=np.float32),
            local_idx=local_idx,
            eta_inner=eta_inner,
        )
        if result is None:
            n_dropped += 1
            continue

        unit = unit_layout_by_id[uid]
        result['meta'] = {
            'unit_id':      uid,
            'unit_type':    getattr(unit, 'unit_type', 'unknown'),
            'layer':        getattr(unit, 'layer', -1),
            'parent_block': block_id,
            'importance':   result['importance'],
        }
        out[uid] = result
        importances.append(result['importance'])

    if verbose:
        print(f"  computed {len(out)} unit subspaces, {n_dropped} dropped")
        if importances:
            arr = np.array(importances)
            print(f"  importance stats: "
                  f"min={arr.min():.3e} median={np.median(arr):.3e} "
                  f"max={arr.max():.3e} ratio max/min={arr.max()/max(arr.min(),1e-12):.1f}")
    return out


# Save / load (compatible with unit_svd.save_task_subspaces format)

def save_task_subspaces(task_subspaces: Dict[Any, Dict[str, Any]], path: str):
    """
    Save weighted subspaces to .npz. Same schema as unit_svd, plus
    importance field.
    """
    keys = list(task_subspaces.keys())
    payload = {'__unit_ids__': np.array(keys, dtype=object)}
    for k in keys:
        r = task_subspaces[k]
        payload[f'{k}__U']           = r['U'].astype(np.float16)
        payload[f'{k}__sigma']       = r['sigma']
        payload[f'{k}__rho']         = np.int32(r['rho'])
        payload[f'{k}__d_k']         = np.int32(r['d_k'])
        payload[f'{k}__var']         = np.float32(r['var_explained'])
        payload[f'{k}__importance']  = np.float32(r['importance'])
        payload[f'{k}__type']        = np.array(r['meta']['unit_type'])
        payload[f'{k}__layer']       = np.int32(r['meta']['layer'])
        payload[f'{k}__parent']      = np.array(r['meta']['parent_block'])
    np.savez_compressed(path, **payload)


def load_task_subspaces(path: str) -> Dict[Any, Dict[str, Any]]:
    """Load .npz produced by save_task_subspaces above."""
    z = np.load(path, allow_pickle=True)
    keys = list(z['__unit_ids__'])
    out = {}
    for k in keys:
        out[k] = {
            'U':            z[f'{k}__U'].astype(np.float32),
            'sigma':        z[f'{k}__sigma'],
            'rho':          int(z[f'{k}__rho']),
            'd_k':          int(z[f'{k}__d_k']),
            'var_explained': float(z[f'{k}__var']),
            'importance':   float(z[f'{k}__importance']),
            'meta': {
                'unit_id':      k,
                'unit_type':    str(z[f'{k}__type']),
                'layer':        int(z[f'{k}__layer']),
                'parent_block': z[f'{k}__parent'].item(),
                'importance':   float(z[f'{k}__importance']),
            },
        }
    return out


# Sigma-weighted importance helpers (keep a sigma-mode npz alongside the
# configured-mode npz so cross-task analysis can use either)

def snapshot_sigma_subs(subs: Dict[Any, Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """
    Copy ``subs`` so its 'importance' fields are independent of the source.

    Call this before any importance_mode override (residual_topk /
    consistency_share), while ``subs[uid]['importance']`` is still the
    ``||M_u||_F²`` from ``compute_task_subspaces_from_blocks``, i.e. the
    sigma_weighted importance.
    """
    return {
        uid: {**v, 'meta': {**v['meta']}}
        for uid, v in subs.items()
    }


def derive_sigma_subs(subs: Dict[Any, Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """
    Recompute sigma_weighted importance from sigma + var_explained, returning
    a new dict with that importance overwriting whatever's there.

    Identity used:
        ||M_u||_F² = sum(s_u²)
                  = sum(sigma_truncated²) / var_explained

    Use this when only a post-override npz (e.g. consistency_share) is
    available.
    """
    out: Dict[Any, Dict[str, Any]] = {}
    for uid, v in subs.items():
        sigma_arr = np.asarray(v['sigma'], dtype=np.float32)
        ve        = float(v['var_explained'])
        sum_s2    = float((sigma_arr ** 2).sum())
        new_imp   = sum_s2 / ve if ve > 0.0 else sum_s2
        nv = {**v, 'meta': {**v['meta']}}
        nv['importance']         = new_imp
        nv['meta']['importance'] = new_imp
        out[uid] = nv
    return out


# Self-test

if __name__ == "__main__":
    """
    Synthetic test:
        - 2 blocks: block_strong (high singular values), block_weak (low)
        - block_strong contains 2 units, block_weak contains 1 unit
        - units in block_strong should have higher importance than the
          unit in block_weak
        - within block_strong, unit_a (rows boosted on the top direction)
          should outrank unit_b
    """
    print("=" * 70)
    print("SYNTHETIC TEST: 2 blocks, 3 units, planted importance gradient")
    print("=" * 70)
    print()

    rng = np.random.default_rng(0)

    class FakeUnit:
        def __init__(self, uid, indices, unit_type, layer):
            self.id = uid
            self._idx = np.asarray(indices, dtype=np.int64)
            self.unit_type = unit_type
            self.layer = layer
        def get_indices(self):
            return self._idx

    # Define units in global parameter space
    # unit_a:  global indices [0..49]    (50 params)
    # unit_b:  global indices [50..149]  (100 params)
    # unit_c:  global indices [150..199] (50 params)
    unit_layout = [
        FakeUnit('unit_a', np.arange(0, 50),    'attn_head', 0),
        FakeUnit('unit_b', np.arange(50, 150),  'attn_head', 0),
        FakeUnit('unit_c', np.arange(150, 200), 'mlp_up',    1),
    ]

    # block_strong: contains unit_a and unit_b
    #   d_block = 50 + 100 = 150
    #   global_indices = [0..149]
    #   sigma values are large (5, 4, 3)
    #   U_B is built so unit_a's rows have more energy than unit_b's in
    #   the top singular direction.
    rho_strong = 3
    d_block_strong = 150
    U_strong = rng.standard_normal((d_block_strong, rho_strong)).astype(np.float32)
    # Boost the first 50 rows (unit_a's territory) on the first column
    U_strong[:50, 0] *= 3.0
    U_strong, _ = np.linalg.qr(U_strong)                # orthonormalize
    sigma_strong = np.array([5.0, 4.0, 3.0], dtype=np.float32)

    # block_weak: contains unit_c
    #   d_block = 50, sigma small (0.5, 0.3)
    rho_weak = 2
    d_block_weak = 50
    U_weak = rng.standard_normal((d_block_weak, rho_weak)).astype(np.float32)
    U_weak, _ = np.linalg.qr(U_weak)
    sigma_weak = np.array([0.5, 0.3], dtype=np.float32)

    block_results = {
        'block_strong': {
            'U':              U_strong,
            'sigma':          sigma_strong,
            'unit_ids':       ['unit_a', 'unit_b'],
            'global_indices': np.arange(0, 150),
        },
        'block_weak': {
            'U':              U_weak,
            'sigma':          sigma_weak,
            'unit_ids':       ['unit_c'],
            'global_indices': np.arange(150, 200),
        },
    }

    subs = compute_task_subspaces_from_blocks(
        block_results=block_results,
        unit_layout=unit_layout,
        eta_inner=0.99,
        verbose=True,
    )

    print()
    print("Per-unit results:")
    print("-" * 70)
    for uid, r in subs.items():
        print(f"  {uid}: parent={r['meta']['parent_block']:>13s}  "
              f"d_k={r['d_k']:3d}  rho={r['rho']:2d}  "
              f"importance={r['importance']:.4f}  "
              f"top_sigma={r['sigma'][0]:.4f}")

    # Verifications
    imp_a = subs['unit_a']['importance']
    imp_b = subs['unit_b']['importance']
    imp_c = subs['unit_c']['importance']

    print()
    print("Verification:")
    print(f"  importance(a) = {imp_a:.4f}")
    print(f"  importance(b) = {imp_b:.4f}")
    print(f"  importance(c) = {imp_c:.4f}")
    print(f"  expect: a, b in strong block >> c in weak block")
    print(f"  expect: a > b (we boosted unit_a's rows on top direction)")

    assert imp_a > imp_c, "unit_a should be more important than unit_c"
    assert imp_b > imp_c, "unit_b should be more important than unit_c"
    assert imp_a > imp_b, "unit_a should outweigh unit_b (we boosted it)"
    print("  OK: all importance orderings correct")

    # Save / load roundtrip
    save_task_subspaces(subs, '/tmp/_weighted_unit_test.npz')
    subs2 = load_task_subspaces('/tmp/_weighted_unit_test.npz')
    for uid in subs:
        max_diff = np.abs(subs[uid]['U'] - subs2[uid]['U']).max()
        imp_diff = abs(subs[uid]['importance'] - subs2[uid]['importance'])
        print(f"  {uid}: roundtrip max diff = {max_diff:.4e}, "
              f"importance diff = {imp_diff:.4e}")
    print("OK")