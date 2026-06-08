"""
distill_patterns.py
-------------------
Collapses the 495 individual crash policies in crash_policies.jsonl into a
compact set of named abstract failure patterns via k-means clustering and
LLM synthesis.

Usage:
    python distill_patterns.py
    python distill_patterns.py --policies crash_policies.jsonl \
        --index policy_index.npz --out abstract_patterns.jsonl \
        --distilled-index distilled_index.npz
    python distill_patterns.py --skip-synthesis   # rebuild index without API calls
    python distill_patterns.py --dry-run          # preview LLM vs curated names, no writes
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
K_VALUES = [15, 20, 25, 30, 35, 40, 50]
CENTROID_TOP_N = 8
CALL_DELAY = 4.0  # 15 RPM free-tier cap → 60/15 = 4s minimum between calls

# All 35 patterns must have entries here to ensure full reproducibility
# across re-runs. If k changes and new pattern IDs are created, add
# curated names for the new IDs before re-running.
PATTERN_NAME_OVERRIDES: dict[str, str] = {
    "P000": "Abrupt Lead Vehicle Deceleration",
    "P001": "Unprotected Left Turn Collision",
    "P002": "Oncoming Traffic on Icy Narrow Roads",
    "P003": "Unexpected Lead Vehicle Stop",
    "P004": "Unsignalized Lateral Lane Encroachment",
    "P005": "Icy Road Lead Vehicle Braking",
    "P006": "Failure to Yield to Pedestrians",
    "P007": "Motorcyclist Following Distance Risk",
    "P008": "Abrupt Lateral Lane Intrusion",
    "P009": "Insufficient Following Distance in Wet Conditions",
    "P010": "Slippery Surface Stopping Distance Deficit",
    "P011": "Unsafe Overtaking Maneuver Conflict",
    "P012": "Insufficient Following Distance at Intersections",
    "P013": "Large Vehicle Encroachment Hazards",
    "P014": "Oncoming Overtaking Encroachment Risk",
    "P015": "Nighttime Intersection Crossing Collision",
    "P016": "Vulnerable Road User Encroachment",
    "P017": "Multi-Lane Congestion Cascade Braking",
    "P018": "Oncoming Passing in Low-Traction Conditions",
    "P019": "Merging Vehicle Speed Adaptation Failure",
    "P020": "Visual Obstruction Induced Risk",
    "P021": "Unsafe Overtaking and Following Dynamics",
    "P022": "Ego-Path Left Turn Yield Failure",
    "P023": "Low Visibility Snow Road Following",
    "P024": "High-Speed Following Distance Deficit",
    "P025": "Large Bus Visibility Obstruction",
    "P026": "Slippery Intersection Encroachment Failure",
    "P027": "Unsafe Lateral Lane Encroachment",
    "P028": "Cross-Traffic Intersection Yield Failure",
    "P029": "Insufficient Following Distance Behind Large Vehicles",
    "P030": "Obstructed Pedestrian Crossing Collision",
    "P031": "Unpredictable Roadside Hazard Encounter",
    "P032": "Insufficient Following Distance in Glare Conditions",
    "P033": "Insufficient Following Distance on Low-Traction Roads",
    "P034": "Side Road Right-of-Way Violation",
}


# ---------------------------------------------------------------------------
# Data loading (mirrors build_index.py)
# ---------------------------------------------------------------------------

def load_policies(jsonl_path: Path) -> tuple[list, list, list, list]:
    triggers, latent_risks, mitigations, clip_ids = [], [], [], []
    required_keys = {"trigger", "latent_risk", "mitigation"}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error") is not None:
                continue
            if not required_keys.issubset(record.keys()):
                continue
            trigger = record["trigger"].strip()
            latent_risk = record["latent_risk"].strip()
            mitigation = record["mitigation"].strip()
            if not trigger or not latent_risk or not mitigation:
                continue
            triggers.append(trigger)
            latent_risks.append(latent_risk)
            mitigations.append(mitigation)
            clip_ids.append(record.get("clip_id", record.get("vidname", "unknown")))
    return triggers, latent_risks, mitigations, clip_ids


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(embeddings: np.ndarray, k_values: list[int], seed: int = 42) -> dict:
    results = {}
    print("\n[cluster] Running k-means for k =", k_values)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(len(embeddings), 2000), random_state=seed)
        results[k] = {"silhouette": float(score), "labels": labels.tolist(), "centers": km.cluster_centers_}
        print(f"  k={k:>3}  silhouette={score:.4f}")
    return results


# ---------------------------------------------------------------------------
# Gemini synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = (
    "You are an expert in autonomous-vehicle safety analysis. "
    "Given a set of crash scenario descriptions, synthesize one canonical abstract failure pattern. "
    "Respond with ONLY a valid JSON object — no preamble, no markdown fences."
)

SYNTHESIS_TEMPLATE = """You are given {n} crash scenario policies that belong to the same risk cluster.
Synthesize them into one canonical abstract failure pattern.

POLICIES:
{policy_block}

Respond with a single JSON object with exactly these four keys:
{{
  "pattern_name": "3-5 word human-readable name, e.g. Insufficient following distance",
  "trigger": "One canonical sentence describing the observable scene conditions that create danger across all these scenarios.",
  "latent_risk": "One canonical sentence describing the failure mode that connects the trigger to a collision.",
  "mitigation": "One canonical sentence describing the corrective action the ego driver should take."
}}"""


def format_policy_block(indices: list[int], triggers: list, latent_risks: list, mitigations: list) -> str:
    lines = []
    for rank, i in enumerate(indices, 1):
        lines.append(
            f"[{rank}] TRIGGER: {triggers[i]}\n"
            f"    LATENT RISK: {latent_risks[i]}\n"
            f"    MITIGATION: {mitigations[i]}"
        )
    return "\n\n".join(lines)


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from:\n{text}")


def _parse_retry_delay(err: Exception) -> float | None:
    """Extract suggested retry_delay seconds from a Gemini 429 error string."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(err))
    return float(m.group(1)) + 2.0 if m else None


def call_gemini(prompt: str, model: genai.GenerativeModel, max_retries: int = 5) -> dict:
    full_prompt = f"{SYNTHESIS_SYSTEM}\n\n{prompt}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(full_prompt)
            return extract_json(response.text.strip())
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str:
                wait = _parse_retry_delay(e) or 60.0
            else:
                wait = 5.0 * (2 ** (attempt - 1))  # 5s, 10s, 20s, ...
            print(f"  [gemini] Error (attempt {attempt}/{max_retries}), retrying in {wait:.0f}s: {err_str[:120]}")
            if attempt < max_retries:
                time.sleep(wait)
    raise RuntimeError(f"Gemini failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Embedding (mirrors build_index.py)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def apply_overrides(patterns: list[dict]) -> int:
    """Apply PATTERN_NAME_OVERRIDES in-place; return count of overrides applied."""
    count = 0
    for p in patterns:
        name = PATTERN_NAME_OVERRIDES.get(p["pattern_id"])
        if name:
            p["pattern_name"] = name
            count += 1
    return count


def save_index(patterns: list[dict], out_path: Path, st_model: SentenceTransformer) -> None:
    pattern_triggers = [p["trigger"] for p in patterns]
    pattern_value_texts = [f"{p['latent_risk']} {p['mitigation']}" for p in patterns]

    print("[distill] Encoding key embeddings (canonical triggers)...")
    distilled_keys = embed_texts(pattern_triggers, st_model)

    print("[distill] Encoding value embeddings (latent_risk + mitigation)...")
    distilled_values = embed_texts(pattern_value_texts, st_model)

    np.savez(
        out_path,
        key_embeddings=distilled_keys,
        value_embeddings=distilled_values,
        triggers=np.array(pattern_triggers, dtype=object),
        latent_risks=np.array([p["latent_risk"] for p in patterns], dtype=object),
        mitigations=np.array([p["mitigation"] for p in patterns], dtype=object),
        vidnames=np.array([p["pattern_id"] for p in patterns], dtype=object),
        pattern_names=np.array([p["pattern_name"] for p in patterns], dtype=object),
        model_name=np.array([MODEL_NAME]),
    )
    print(f"[distill] Distilled index saved to: {out_path}")


def print_reproducibility(seed: int, best_k: int, best_sil: float,
                          n_policies: int, n_patterns: int, n_overrides: int) -> None:
    compression = n_policies / n_patterns if n_patterns else 0.0
    print("\n=== REPRODUCIBILITY ===")
    print(f"Random seed : {seed}")
    print(f"Best k      : {best_k}")
    print(f"Silhouette  : {best_sil:.4f}")
    print(f"Policies    : {n_policies}")
    print(f"Patterns    : {n_patterns}")
    print(f"Compression : {compression:.1f}x")
    print(f"Overrides   : {n_overrides}/{n_patterns} applied")
    print("=======================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distill crash policies into abstract failure patterns")
    parser.add_argument("--policies", type=Path, default=Path("crash_policies.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("policy_index.npz"))
    parser.add_argument("--out", type=Path, default=Path("abstract_patterns.jsonl"))
    parser.add_argument("--distilled-index", type=Path, default=Path("distilled_index.npz"))
    parser.add_argument("--clustering-results", type=Path, default=Path("clustering_results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-synthesis", action="store_true",
        help="Skip Gemini API calls entirely. Load existing abstract_patterns.jsonl, "
             "re-apply overrides, and rebuild distilled_index.npz.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run clustering and synthesis, print LLM vs curated names, write no output files.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # --skip-synthesis: load existing patterns, reapply overrides, rebuild index
    # -----------------------------------------------------------------------
    if args.skip_synthesis:
        print("[distill] --skip-synthesis: loading existing patterns from:", args.out)
        patterns = [json.loads(l) for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_overrides = apply_overrides(patterns)
        print(f"[distill] Applied {n_overrides} curated name overrides.")

        args.out.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in patterns) + "\n",
            encoding="utf-8",
        )
        print(f"[distill] Saved updated patterns to: {args.out}")

        print(f"\n[distill] Building distilled index with {MODEL_NAME}...")
        st_model = SentenceTransformer(MODEL_NAME)
        save_index(patterns, args.distilled_index, st_model)

        n_policies = sum(p["cluster_size"] for p in patterns)
        print_reproducibility(
            seed=args.seed, best_k=len(patterns), best_sil=float("nan"),
            n_policies=n_policies, n_patterns=len(patterns), n_overrides=n_overrides,
        )
        return

    # -----------------------------------------------------------------------
    # Normal / dry-run path: load policies and embeddings, cluster, synthesize
    # -----------------------------------------------------------------------
    api_key = os.getenv("GEM_KEY03")
    if not api_key:
        raise EnvironmentError("GEM_KEY03 not set. Add it to .env.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    # ---- Step 1: Load policies and embeddings ----
    print("[distill] Loading policies from:", args.policies)
    triggers, latent_risks, mitigations, clip_ids = load_policies(args.policies)
    n_original = len(triggers)
    print(f"[distill] Loaded {n_original} valid triplet policies.")

    print("[distill] Loading trigger embeddings from:", args.index)
    npz = np.load(args.index, allow_pickle=True)
    key_embeddings = npz["key_embeddings"].astype(np.float32)

    if len(key_embeddings) != n_original:
        print(
            f"[distill] WARNING: index has {len(key_embeddings)} entries but JSONL has {n_original}. "
            "Using index order."
        )
        n = min(len(key_embeddings), n_original)
        key_embeddings = key_embeddings[:n]
        triggers = triggers[:n]
        latent_risks = latent_risks[:n]
        mitigations = mitigations[:n]
        clip_ids = clip_ids[:n]

    n_clustered = len(key_embeddings)

    # ---- Step 2: K-means sweep ----
    cluster_data = run_clustering(key_embeddings, K_VALUES, seed=args.seed)

    print(f"\n{'k':>5}  {'silhouette':>12}")
    print("-" * 20)
    for k in K_VALUES:
        print(f"{k:>5}  {cluster_data[k]['silhouette']:>12.4f}")

    clustering_out = {
        str(k): {"k": k, "silhouette": cluster_data[k]["silhouette"]}
        for k in K_VALUES
    }
    if not args.dry_run:
        with open(args.clustering_results, "w") as f:
            json.dump(clustering_out, f, indent=2)
        print(f"\n[distill] Clustering results saved to: {args.clustering_results}")

    best_k = max(K_VALUES, key=lambda k: cluster_data[k]["silhouette"])
    best_sil = cluster_data[best_k]["silhouette"]
    print(f"[distill] Best k={best_k} (silhouette={best_sil:.4f})")

    labels = np.array(cluster_data[best_k]["labels"])
    centers = cluster_data[best_k]["centers"]

    # ---- Step 3: Synthesize one pattern per cluster ----
    if args.dry_run:
        print(f"\n[distill] --dry-run: synthesizing {best_k} patterns (no files will be written)...")
    else:
        print(f"\n[distill] Synthesizing {best_k} abstract patterns via Gemini ({GEMINI_MODEL})...")

    patterns = []

    for cluster_id in tqdm(range(best_k), desc="Synthesizing clusters"):
        member_indices = np.where(labels == cluster_id)[0]
        centroid = centers[cluster_id]

        member_embs = key_embeddings[member_indices]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = member_embs @ centroid_norm
        top_idx = np.argsort(sims)[::-1][: min(CENTROID_TOP_N, len(member_indices))]
        closest_global = member_indices[top_idx].tolist()

        policy_block = format_policy_block(closest_global, triggers, latent_risks, mitigations)
        prompt = SYNTHESIS_TEMPLATE.format(n=len(closest_global), policy_block=policy_block)

        result = call_gemini(prompt, gemini_model)

        pid = f"P{cluster_id:03d}"
        llm_name = result.get("pattern_name", f"Pattern {cluster_id}")

        if args.dry_run:
            curated = PATTERN_NAME_OVERRIDES.get(pid)
            if curated:
                print(f'[{pid}] LLM: "{llm_name}" → CURATED: "{curated}"')
            else:
                print(f'[{pid}] LLM: "{llm_name}"')

        patterns.append(
            {
                "pattern_id": pid,
                "pattern_name": llm_name,
                "trigger": result.get("trigger", ""),
                "latent_risk": result.get("latent_risk", ""),
                "mitigation": result.get("mitigation", ""),
                "cluster_size": int(len(member_indices)),
                "source_clip_ids": [clip_ids[i] for i in member_indices.tolist()],
            }
        )

        if cluster_id < best_k - 1:
            time.sleep(CALL_DELAY)

    if args.dry_run:
        print("\n[distill] --dry-run complete. No files written.")
        print_reproducibility(
            seed=args.seed, best_k=best_k, best_sil=best_sil,
            n_policies=n_clustered, n_patterns=len(patterns),
            n_overrides=sum(1 for p in patterns if p["pattern_id"] in PATTERN_NAME_OVERRIDES),
        )
        return

    # ---- Step 4: Apply curated name overrides ----
    n_overrides = apply_overrides(patterns)
    print(f"[distill] Applied {n_overrides} curated name overrides.")

    # ---- Step 5: Save abstract_patterns.jsonl ----
    with open(args.out, "w", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p) + "\n")
    print(f"[distill] Abstract patterns saved to: {args.out}")

    # ---- Step 6: Build distilled index ----
    print(f"\n[distill] Building distilled index with {MODEL_NAME}...")
    st_model = SentenceTransformer(MODEL_NAME)
    save_index(patterns, args.distilled_index, st_model)

    # ---- Summary ----
    cluster_sizes = [p["cluster_size"] for p in patterns]
    print("\n" + "=" * 55)
    print("DISTILLATION SUMMARY")
    print("=" * 55)
    print(f"  Original corpus size : {n_original}")
    print(f"  Abstract patterns    : {len(patterns)}")
    print(f"  Compression ratio    : {n_original / len(patterns):.1f}x")
    print(f"  Cluster sizes        : min={min(cluster_sizes)}  max={max(cluster_sizes)}  mean={np.mean(cluster_sizes):.1f}")
    print("\n  PATTERN NAMES:")
    for p in patterns:
        print(f"    [{p['pattern_id']}] {p['pattern_name']}  (n={p['cluster_size']})")
    print("=" * 55)

    print_reproducibility(
        seed=args.seed, best_k=best_k, best_sil=best_sil,
        n_policies=n_clustered, n_patterns=len(patterns), n_overrides=n_overrides,
    )


if __name__ == "__main__":
    main()
