#!/usr/bin/env python3
"""
Normalized Voice Fidelity Analysis
====================================
CDS529 — Comparing LLM speech to real character speech, normalized by baseline.

METHOD:
  For each character:
    1. V_all  = Empath vector of ALL extracted dialogue (full linguistic baseline)
    2. V_real = Empath vector of the specific real lines (context-response pairs)
    3. V_llm  = Empath vector of the LLM-generated responses

  Normalize:
    4. R_real = V_real / V_all  (how real lines deviate from character's typical speech)
    5. R_llm  = V_llm / V_all  (how LLM lines deviate from character's typical speech)

  Compare:
    6. Gap = cosine_similarity(R_real, R_llm) + MAE(R_real, R_llm)

  If both ratios are similar, the LLM captures the character's voice accurately.

Usage:
    pip install empath pandas numpy matplotlib
    python analyze_normalized_voice.py --results ./voice_fidelity_results

    OR if you have a different results dir:
    python analyze_normalized_voice.py --results ./voice_pdb100_results
"""

import os
import sys
import json
import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from empath import Empath
except ImportError:
    print("[ERROR] pip install empath")
    sys.exit(1)

lexicon = Empath()
CATEGORIES = list(lexicon.analyze("test", normalize=True).keys())
N_CATS = len(CATEGORIES)
EPSILON = 1e-8  # avoid division by zero


# ╔══════════════════════════════════════════════════════════════╗
# ║  EMPATH HELPERS                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def text_to_vector(text: str) -> np.ndarray:
    """Convert text to Empath feature vector (194 categories)."""
    if not text or not text.strip():
        return np.zeros(N_CATS)
    scores = lexicon.analyze(text, normalize=True)
    if not scores:
        return np.zeros(N_CATS)
    return np.array([scores.get(c, 0.0) for c in CATEGORIES], dtype=float)


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def mae(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.mean(np.abs(v1 - v2)))


# ╔══════════════════════════════════════════════════════════════╗
# ║  LOAD DATA FROM PIPELINE OUTPUT                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_pipeline_data(results_dir: str) -> dict:
    """
    Load all necessary data from the voice fidelity pipeline output.
    Returns dict with:
      - transcript_texts: {char_name: full_transcript_text}
      - canon_lines: {char_name: text of real lines from pairs}
      - llm_lines: {(char_name, model_key): text of LLM responses}
      - all_extracted_lines: {char_name: text of ALL extracted character lines}
    """
    canon_dir = os.path.join(results_dir, "liwc_ready", "canon_lines")
    llm_dir = os.path.join(results_dir, "liwc_ready", "llm_responses")
    ckpt_path = os.path.join(results_dir, "checkpoint.json")

    data = {
        'canon_lines': {},
        'llm_lines': {},
        'all_extracted_lines': {},
    }

    # ── First, load checkpoint to build name mapping ──
    # Checkpoint uses ORIGINAL names like "Anton Chigurh"
    # Files use SANITIZED names like "Anton_Chigurh"
    # We need a reverse mapping: sanitized -> original
    sanitized_to_original = {}

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)

        for key, value in ckpt.items():
            if key.startswith('pairs:'):
                original_name = key[len('pairs:'):]
                sanitized = re.sub(r'[^\w\-]', '_', original_name)
                sanitized_to_original[sanitized] = original_name

                # Also build all_extracted_lines from checkpoint
                if isinstance(value, list):
                    all_lines = [p.get('real_line', '') for p in value if isinstance(p, dict)]
                    full_text = '\n'.join(line for line in all_lines if line)
                    if full_text:
                        data['all_extracted_lines'][original_name] = full_text
    else:
        print("  [WARN] No checkpoint.json found — baseline data unavailable")

    def resolve_name(sanitized: str) -> str:
        """Map a sanitized filename back to the original character name."""
        if sanitized in sanitized_to_original:
            return sanitized_to_original[sanitized]
        # Fallback: just return the sanitized version
        return sanitized

    # ── Load canon lines ──
    if os.path.exists(canon_dir):
        for fname in sorted(os.listdir(canon_dir)):
            if fname.endswith('_canon.txt'):
                sanitized = fname.replace('_canon.txt', '')
                original = resolve_name(sanitized)
                fpath = os.path.join(canon_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    data['canon_lines'][original] = text

    # ── Load LLM response lines ──
    # Filenames: "CharName_sanitized_modelkey.txt"
    # Problem: model_key is after the LAST underscore, but char name has underscores too
    # Solution: try all known model keys as suffix
    known_models = ['claude', 'chatgpt', 'gemini', 'characterglm', 'deepseek']

    if os.path.exists(llm_dir):
        for fname in sorted(os.listdir(llm_dir)):
            if not fname.endswith('.txt'):
                continue
            base = fname.replace('.txt', '')

            # Try each known model as suffix
            matched = False
            for mk in known_models:
                suffix = f'_{mk}'
                if base.endswith(suffix):
                    sanitized = base[:-len(suffix)]
                    original = resolve_name(sanitized)
                    fpath = os.path.join(llm_dir, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        data['llm_lines'][(original, mk)] = text
                    matched = True
                    break

            if not matched:
                # Fallback: rsplit on last underscore
                parts = base.rsplit('_', 1)
                if len(parts) == 2:
                    sanitized, mk = parts
                    original = resolve_name(sanitized)
                    fpath = os.path.join(llm_dir, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        data['llm_lines'][(original, mk)] = text

    return data


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN ANALYSIS                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def run_analysis(results_dir: str):
    print(f"\n{'='*70}")
    print(f" Normalized Voice Fidelity Analysis")
    print(f" Method: (V_llm / V_all) vs (V_real / V_all)")
    print(f"{'='*70}\n")

    data = load_pipeline_data(results_dir)

    print(f"  Canon line files:       {len(data['canon_lines'])}")
    print(f"  LLM response files:     {len(data['llm_lines'])}")
    print(f"  Characters with baseline: {len(data['all_extracted_lines'])}")

    if not data['canon_lines'] or not data['llm_lines']:
        print("[ERROR] No data found. Check results directory.")
        return

    # ── Compute vectors ──
    print(f"\n  Computing Empath vectors...")

    # V_all: baseline vector per character (ALL extracted dialogue)
    v_all = {}
    for char_name, text in data['all_extracted_lines'].items():
        v = text_to_vector(text)
        if np.sum(v) > 0:
            v_all[char_name] = v

    # V_real: vector of the specific real lines used in pairs
    v_real = {}
    for char_name, text in data['canon_lines'].items():
        v = text_to_vector(text)
        if np.sum(v) > 0:
            v_real[char_name] = v

    # V_llm: vector of LLM-generated responses
    v_llm = {}
    for (char_name, model_key), text in data['llm_lines'].items():
        v = text_to_vector(text)
        if np.sum(v) > 0:
            v_llm[(char_name, model_key)] = v

    # ── Normalize and compare ──
    print(f"  Computing normalized ratios and gaps...\n")

    # Minimum word count for meaningful Empath analysis
    MIN_WORDS = 30

    results = []
    skipped_low_words = 0

    for (char_name, model_key), llm_vec in v_llm.items():
        if char_name not in v_all or char_name not in v_real:
            continue

        # Skip characters with too few words for reliable Empath
        baseline_words = len(data['all_extracted_lines'].get(char_name, '').split())
        if baseline_words < MIN_WORDS:
            skipped_low_words += 1
            continue

        baseline = v_all[char_name]
        real_vec = v_real[char_name]

        # MASK: only keep categories where the baseline has non-zero values
        # If a character never uses "children" words, comparing that is meaningless
        nonzero_mask = baseline > 0
        n_active = np.sum(nonzero_mask)

        if n_active < 5:
            # Too few active categories for meaningful comparison
            skipped_low_words += 1
            continue

        # Extract only the active categories
        b_active = baseline[nonzero_mask]
        r_active = real_vec[nonzero_mask]
        l_active = llm_vec[nonzero_mask]

        # Normalize: ratio relative to baseline (element-wise)
        r_real_norm = r_active / b_active
        r_llm_norm = l_active / b_active

        # Compare the two ratio vectors
        cos = cosine_sim(r_real_norm, r_llm_norm)
        gap_mae = mae(r_real_norm, r_llm_norm)

        # Also compute raw (unnormalized) cosine for comparison
        raw_cos = cosine_sim(real_vec, llm_vec)

        results.append({
            'character': char_name,
            'model': model_key,
            'normalized_cosine_sim': round(cos, 4),
            'normalized_mae': round(gap_mae, 4),
            'raw_cosine_sim': round(raw_cos, 4),
            'active_categories': int(n_active),
            'baseline_word_count': baseline_words,
            'real_word_count': len(data['canon_lines'].get(char_name, '').split()),
            'llm_word_count': len(data['llm_lines'].get((char_name, model_key), '').split()),
        })

    if skipped_low_words:
        print(f"  Skipped {skipped_low_words} comparisons (baseline < {MIN_WORDS} words or < 5 active categories)")

    if not results:
        print("[ERROR] No valid comparisons could be made.")
        return

    df = pd.DataFrame(results)

    # ════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════

    print(f"{'='*70}")
    print(f" RESULTS ({len(df)} comparisons)")
    print(f"{'='*70}")

    # ── Per-model ranking (normalized) ──
    print(f"\n--- Per-Model Ranking (Normalized Cosine Similarity) ---")
    print(f"    Higher = LLM deviates from baseline in the same way as the real lines\n")
    model_avg = df.groupby('model').agg({
        'normalized_cosine_sim': ['mean', 'std'],
        'normalized_mae': 'mean',
        'raw_cosine_sim': 'mean',
    }).round(4)
    model_avg.columns = ['Norm Cosine (mean)', 'Norm Cosine (std)', 'Norm MAE (mean)', 'Raw Cosine (mean)']
    model_avg = model_avg.sort_values('Norm Cosine (mean)', ascending=False)
    print(model_avg.to_string())

    # ── Per-character ranking ──
    print(f"\n--- Per-Character (averaged across models) ---\n")
    char_avg = df.groupby('character').agg({
        'normalized_cosine_sim': 'mean',
        'normalized_mae': 'mean',
        'raw_cosine_sim': 'mean',
    }).round(4)
    char_avg.columns = ['Norm Cosine', 'Norm MAE', 'Raw Cosine']
    char_avg = char_avg.sort_values('Norm Cosine', ascending=False)
    print(char_avg.to_string())

    # ── Full matrix ──
    print(f"\n--- Normalized Cosine Similarity Matrix (Character × Model) ---\n")
    pivot = df.pivot_table(index='character', columns='model', values='normalized_cosine_sim').round(4)
    print(pivot.to_string())

    # ── Best model per character ──
    print(f"\n--- Best Model per Character (Normalized) ---\n")
    for char in pivot.index:
        row = pivot.loc[char].dropna()
        if len(row) > 0:
            best = row.idxmax()
            val = row.max()
            print(f"  {char:<40s}  {best:15s}  {val:.4f}")

    # ── Win count ──
    print(f"\n--- Win Count ---\n")
    wins = {}
    for char in pivot.index:
        row = pivot.loc[char].dropna()
        if len(row) > 0:
            best = row.idxmax()
            wins[best] = wins.get(best, 0) + 1
    for model, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {model:20s}: {count} characters")

    # ── Top divergent categories ──
    print(f"\n--- Top Empath Categories Where LLMs Diverge Most (Normalized) ---\n")
    cat_diffs = {c: [] for c in CATEGORIES}
    for (char_name, model_key), llm_vec in v_llm.items():
        if char_name not in v_all or char_name not in v_real:
            continue
        baseline = v_all[char_name]
        baseline_words = len(data['all_extracted_lines'].get(char_name, '').split())
        if baseline_words < MIN_WORDS:
            continue
        for ci, cat in enumerate(CATEGORIES):
            if baseline[ci] > 0:  # only compare active categories
                r_real_cat = v_real[char_name][ci] / baseline[ci]
                r_llm_cat = llm_vec[ci] / baseline[ci]
                cat_diffs[cat].append(r_llm_cat - r_real_cat)

    avg_diffs = {cat: (np.mean(vals), np.mean(np.abs(vals))) for cat, vals in cat_diffs.items() if vals}
    sorted_cats = sorted(avg_diffs.items(), key=lambda x: x[1][1], reverse=True)

    print(f"  {'Category':<30s}  {'Avg |Gap|':>10s}  {'Direction':>12s}")
    print(f"  {'─'*56}")
    for cat, (signed, unsigned) in sorted_cats[:20]:
        direction = "LLM overuses ↑" if signed > 0 else "LLM underuses ↓"
        print(f"  {cat:<30s}  {unsigned:>10.4f}  {direction}")

    # ── Improvement from normalization ──
    print(f"\n--- Normalization Effect ---")
    print(f"    Does normalizing by baseline change the results?\n")
    for model in sorted(df['model'].unique()):
        sub = df[df['model'] == model]
        raw_mean = sub['raw_cosine_sim'].mean()
        norm_mean = sub['normalized_cosine_sim'].mean()
        delta = norm_mean - raw_mean
        print(f"  {model:20s}: raw={raw_mean:.4f}  norm={norm_mean:.4f}  delta={delta:+.4f}")

    # ── Save results ──
    output_csv = os.path.join(results_dir, "normalized_voice_fidelity.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nResults saved: {output_csv}")

    matrix_csv = os.path.join(results_dir, "normalized_similarity_matrix.csv")
    pivot.to_csv(matrix_csv, encoding='utf-8')
    print(f"Matrix saved:  {matrix_csv}")

    # ── Charts ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Chart 1: Model ranking — normalized vs raw
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Normalized
        ax = axes[0]
        model_means = df.groupby('model')['normalized_cosine_sim'].mean().sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_means)))
        bars = ax.barh(model_means.index, model_means.values, color=colors)
        ax.set_xlabel('Normalized Cosine Similarity')
        ax.set_title('Normalized (V/V_all)')
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, model_means.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

        # Raw
        ax = axes[1]
        model_means_raw = df.groupby('model')['raw_cosine_sim'].mean().sort_values(ascending=True)
        bars = ax.barh(model_means_raw.index, model_means_raw.values, color=colors)
        ax.set_xlabel('Raw Cosine Similarity')
        ax.set_title('Raw (no normalization)')
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, model_means_raw.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

        fig.suptitle('LLM Voice Fidelity: Normalized vs Raw', fontsize=13)
        plt.tight_layout()
        chart1 = os.path.join(results_dir, "chart_normalized_vs_raw_ranking.png")
        fig.savefig(chart1, dpi=150)
        plt.close(fig)
        print(f"Chart saved: {chart1}")

        # Chart 2: Heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if val < 0.6 else 'black')
        fig.colorbar(im, ax=ax, label='Normalized Cosine Similarity', shrink=0.8)
        ax.set_title('Normalized Voice Fidelity (Character × LLM)')
        plt.tight_layout()
        chart2 = os.path.join(results_dir, "chart_normalized_heatmap.png")
        fig.savefig(chart2, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Chart saved: {chart2}")

        # Chart 3: Per-category divergence (top 15)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_cats = sorted_cats[:15]
        cat_names = [c[0] for c in top_cats]
        cat_signed = [c[1][0] for c in top_cats]
        colors_bar = ['#d32f2f' if v > 0 else '#1976d2' for v in cat_signed]
        ax.barh(range(len(cat_names)), cat_signed, color=colors_bar)
        ax.set_yticks(range(len(cat_names)))
        ax.set_yticklabels(cat_names, fontsize=9)
        ax.set_xlabel('Mean Normalized Gap (positive = LLM overuses)')
        ax.set_title('Top 15 Empath Categories: LLM Deviation from Real Speech Pattern')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        chart3 = os.path.join(results_dir, "chart_category_divergence.png")
        fig.savefig(chart3, dpi=150)
        plt.close(fig)
        print(f"Chart saved: {chart3}")

    except ImportError:
        print("[WARN] matplotlib not installed, skipping charts")

    print(f"\n{'='*70}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalized Voice Fidelity Analysis")
    parser.add_argument("--results", default="./voice_fidelity_results",
                        help="Voice fidelity results directory")
    args = parser.parse_args()
    if not os.path.exists(args.results):
        print(f"[ERROR] Results dir not found: {args.results}")
        sys.exit(1)
    run_analysis(args.results)
