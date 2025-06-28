import pandas as pd
from typing import Dict

def drift_report_node(
    drift_flags: Dict[str, bool],
    drift_report: pd.DataFrame
) -> str:
    alerts = [f for f, flag in drift_flags.items() if flag]

    if alerts:
        print(f"[DRIFT ALERT] Features with drift: {', '.join(alerts)}")
    else:
        print("[DRIFT CHECK] No drift detected.")

    lines = []
    lines.append("DATA DRIFT REPORT\n")
    
    lines.append(f"{'Feature':<25} {'KL Divergence':>15} {'JS Divergence':>15} {'Drift Detected':>15}\n")
    lines.append("-" * 80 + "\n")
    
    for _, row in drift_report.iterrows():
        lines.append(f"{row['feature']:<25} {row['kl_divergence']:>15.6f} {row['js_divergence']:>15.6f} {str(row['drift_flag']):>15}\n")

    lines.append("\n" + "-" * 80 + "\n")
    lines.append("SUMMARY\n")
    
    total_features = len(drift_report)
    drifted_features = drift_report['drift_flag'].sum()
    drift_percentage = (drifted_features / total_features * 100) if total_features > 0 else 0
    
    lines.append(f"Total features analyzed: {total_features}\n")
    lines.append(f"Features with drift detected: {drifted_features}\n")
    lines.append(f"Drift percentage: {drift_percentage:.1f}%\n")
    
    avg_kl = drift_report['kl_divergence'].mean()
    avg_js = drift_report['js_divergence'].mean()
    lines.append(f"Average KL Divergence: {avg_kl:.6f}\n")
    lines.append(f"Average JS Divergence: {avg_js:.6f}\n")

    if drifted_features == 0:
        status = "NO DRIFT DETECTED"
    elif drift_percentage <= 20:
        status = "LOW DRIFT"
    elif drift_percentage <= 50:
        status = "MODERATE DRIFT"
    else:
        status = "HIGH DRIFt"
    
    lines.append(f"\nStatus: {status}\n")
    
    if alerts:
        lines.append(f"\nFeatures requiring attention: {', '.join(alerts)}\n")

    return "".join(lines)