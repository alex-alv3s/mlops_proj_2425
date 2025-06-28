import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

def normalize_distribution(arr: np.ndarray, bins: int = 20) -> np.ndarray:
    hist, _ = np.histogram(arr, bins=bins, range=(np.nanmin(arr), np.nanmax(arr)), density=True)
    hist += 1e-8
    hist /= hist.sum()
    return hist

def normalize_binary_distribution(arr: np.ndarray) -> np.ndarray:
    counts = pd.Series(arr).value_counts(normalize=True).reindex([0.0, 1.0], fill_value=0.0)
    dist = counts.values + 1e-8
    return dist / dist.sum()

def drift_detection(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    js_threshold: float = 0.1,
    kl_threshold: float = 0.1,
    bins: int = 20
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, bool], pd.DataFrame]:
    
    drift_metrics = {}
    drift_flags = {}
    drift_rows = []

    oh_cols = [col for col in reference.columns if col.startswith('oh_')]
    numeric_cols = [col for col in reference.columns if col not in oh_cols]

    for col in numeric_cols + oh_cols:
        ref_data = reference[col].dropna()
        batch_data = batch[col].dropna()

        if col in oh_cols:
            ref_dist = normalize_binary_distribution(ref_data)
            batch_dist = normalize_binary_distribution(batch_data)
        else:
            ref_dist = normalize_distribution(ref_data, bins)
            batch_dist = normalize_distribution(batch_data, bins)

        kl_div = float(entropy(ref_dist, batch_dist))
        js_div = float(jensenshannon(ref_dist, batch_dist) ** 2)

        drift_metrics[col] = {"kl_divergence": kl_div, "js_divergence": js_div}
        drift_flags[col] = js_div > js_threshold or kl_div > kl_threshold

        drift_rows.append({
            "feature": col,
            "kl_divergence": kl_div,
            "js_divergence": js_div,
            "drift_flag": drift_flags[col]
        })

    drift_report = pd.DataFrame(drift_rows)
    logger.info(f"Drift detection completed on {len(numeric_cols + oh_cols)} features.")
    
    return drift_metrics, drift_flags, drift_report