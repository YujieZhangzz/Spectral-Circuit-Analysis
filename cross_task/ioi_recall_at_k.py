"""
Recall@K of cross-task unit rankings against the 26 canonical IOI heads
(Wang et al.).

For each importance variant under cross_task/run_gpt_12tasks/, ranks attention
heads by score_min, score_mean, score_max, n_significant and importance, and
writes ioi_recall_at_k.csv and ioi_recall_at_k.txt (GT heads recovered and
their ranks).

Unit layout (GPT-2 small): unit_id = 14*layer + head for heads 0..11,
14*layer+12 = mlp_gate, 14*layer+13 = mlp_down.

Usage:
    python cross_task/ioi_recall_at_k.py
"""
import csv
import pickle
from pathlib import Path

ROOT = Path("cross_task/run_gpt_12tasks")
VARIANTS = ["consistency_share", "sigma_weighted"]

# 26 canonical IOI heads by role
IOI_HEADS_BY_ROLE = {
    "Previous Token":      [(2, 2), (4, 11)],
    "Duplicate Token":     [(0, 1), (0, 10), (3, 0)],
    "Induction":           [(5, 5), (5, 8), (5, 9), (6, 9)],
    "S-Inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Name Mover":          [(9, 6), (9, 9), (10, 0)],
    "Negative Name Mover": [(10, 7), (11, 10)],
    "Backup Name Mover":   [(9, 0), (9, 7), (10, 1), (10, 2),
                            (10, 6), (10, 10), (11, 2), (11, 9)],
}


def head_to_unit_id(layer, head_idx):
    return 14 * layer + head_idx


def unit_id_to_label(unit_id):
    layer = unit_id // 14
    rem = unit_id % 14
    if rem < 12:
        return f"L{layer}H{rem}"
    return f"L{layer}_MLP{'gate' if rem == 12 else 'down'}"


IOI_HEAD_TO_ROLE = {}
IOI_GT_UNIT_IDS = set()
for role, heads in IOI_HEADS_BY_ROLE.items():
    for layer, h in heads:
        uid = head_to_unit_id(layer, h)
        IOI_HEAD_TO_ROLE[uid] = role
        IOI_GT_UNIT_IDS.add(uid)
assert len(IOI_GT_UNIT_IDS) == 26, f"GT count = {len(IOI_GT_UNIT_IDS)}"

# Sanity: IOI heads must all map to attn_head positions (rem < 12)
for uid in IOI_GT_UNIT_IDS:
    assert uid % 14 < 12, f"unit_id {uid} not an attn_head slot"


def build_rows(result):
    """Same rows as enrich_universal_units.py, with importance pulled from meta."""
    unit_meta = result["tensor"]["unit_meta"]
    u = result["universality"]
    rows = []
    for i, m in enumerate(unit_meta):
        rows.append({
            "unit_id":       m["unit_id"],
            "unit_type":     m["unit_type"],
            "layer":         m["layer"],
            "importance":    float(m.get("importance", 0.0)),
            "score_min":     float(u["min"][i]),
            "score_mean":    float(u["mean"][i]),
            "score_max":     float(u["max"][i]),
            "n_significant": int(u["n_significant"][i]),
        })
    return rows


def recall_at_k(ranked_unit_ids, gt_set, k):
    """Fraction of GT items appearing in top-k."""
    top = set(ranked_unit_ids[:k])
    return len(top & gt_set) / len(gt_set)


def random_baseline_recall(k, n_attn=144, n_gt=26):
    """E[recall@K] for uniform-random ranking over n_attn attn_head units."""
    return k / n_attn


def evaluate_variant(variant, vdir):
    result = pickle.loads((vdir / "cross_task_result.pkl").read_bytes())
    rows = build_rows(result)
    attn_rows = [r for r in rows if r["unit_type"] == "attn_head"]

    rankings = {
        "score_min":     sorted(attn_rows, key=lambda r: -r["score_min"]),
        "score_mean":    sorted(attn_rows, key=lambda r: -r["score_mean"]),
        "score_max":     sorted(attn_rows, key=lambda r: -r["score_max"]),
        "n_significant": sorted(attn_rows, key=lambda r: (-r["n_significant"], -r["score_mean"])),
        "importance":    sorted(attn_rows, key=lambda r: -r["importance"]),
    }

    K_LIST = [10, 20, 26, 30, 50]
    csv_rows, text_lines = [], []
    text_lines.append(f"=== {variant} ===")
    text_lines.append(f"GT: 26 IOI canonical heads (attn-only ranking)")
    text_lines.append(f"Random baseline: recall@K = K/144")
    text_lines.append("")

    for rname, ranked in rankings.items():
        ranked_ids = [r["unit_id"] for r in ranked]
        text_lines.append(f"--- ranking by {rname} ---")
        text_lines.append(f"top-30 unit_ids: {ranked_ids[:30]}")
        # which GT heads appear and at what rank
        hits = []
        for rank, uid in enumerate(ranked_ids, 1):
            if uid in IOI_GT_UNIT_IDS:
                hits.append((rank, uid, unit_id_to_label(uid), IOI_HEAD_TO_ROLE[uid]))
        text_lines.append(f"GT heads recovered (at any rank): {len(hits)}/26")
        for rank, uid, label, role in hits[:30]:
            text_lines.append(f"  rank={rank:>3}  uid={uid:>3}  {label:<7}  ({role})")
        for K in K_LIST:
            r = recall_at_k(ranked_ids, IOI_GT_UNIT_IDS, K)
            base = random_baseline_recall(K)
            lift = r / base if base > 0 else float("nan")
            text_lines.append(
                f"  recall@{K:>2} = {r:.3f}  ({int(round(r*26))}/26 GT)   "
                f"baseline={base:.3f}   lift={lift:.2f}x"
            )
            csv_rows.append({
                "variant": variant, "ranking": rname, "K": K,
                "recall": f"{r:.4f}",
                "n_hits": int(round(r * 26)),
                "baseline": f"{base:.4f}",
                "lift_over_baseline": f"{lift:.3f}",
            })
        text_lines.append("")

    return csv_rows, text_lines


def main():
    all_csv_rows = []
    all_text = []
    for variant in VARIANTS:
        vdir = ROOT / variant
        pkl = vdir / "cross_task_result.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} missing")
            continue
        csv_rows, text_lines = evaluate_variant(variant, vdir)
        all_csv_rows.extend(csv_rows)
        all_text.extend(text_lines)

        (vdir / "ioi_recall_at_k.txt").write_text("\n".join(text_lines) + "\n")
        with open(vdir / "ioi_recall_at_k.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[wrote] {vdir}/ioi_recall_at_k.csv")
        print(f"[wrote] {vdir}/ioi_recall_at_k.txt")

    # Summary table to stdout
    print("\n" + "=" * 80)
    print("Cross-variant summary (recall):")
    print("=" * 80)
    print("\n".join(all_text))


if __name__ == "__main__":
    main()