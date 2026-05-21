from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_PATH = DATA_DIR / "du_data.csv"
ANOMALY_OUTPUT_PATH = OUTPUT_DIR / "du_anomaly_insights.csv"


def simulate_du_data(n: int = 5000, anomaly_count: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = pd.DataFrame(
        {
            "device_id": np.arange(n),
            "dma": rng.choice(["NY", "LA", "TX", "FL"], n),
            "device_type": rng.choice(["A", "B", "C"], n),
            "hour": rng.integers(0, 24, n),
            "content_type": rng.choice(["sports", "news", "movie"], n),
            "ad_load": rng.uniform(0, 1, n),
        }
    )

    data["disk_utilization"] = 30 + data["hour"] * 1.2 + data["ad_load"] * 40

    anomaly_idx = rng.choice(n, anomaly_count, replace=False)
    data.loc[anomaly_idx, "disk_utilization"] += 40
    data["is_injected_anomaly"] = False
    data.loc[anomaly_idx, "is_injected_anomaly"] = True

    return data


def engineer_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = data.copy()

    scaler = StandardScaler()
    model_data["du_scaled"] = scaler.fit_transform(model_data[["disk_utilization"]])

    encoded_data = pd.get_dummies(
        model_data,
        columns=["dma", "device_type", "content_type"],
        dtype=int,
    )

    feature_columns = [
        column
        for column in encoded_data.columns
        if column not in {"device_id", "disk_utilization", "is_injected_anomaly"}
    ]

    return encoded_data, encoded_data[feature_columns]


def detect_anomalies(data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    model = IsolationForest(contamination=0.02, random_state=42)
    result = data.copy()
    result["anomaly"] = model.fit_predict(features)
    result["anomaly_score"] = model.decision_function(features)
    return result


def explain(row: pd.Series) -> str:
    reason = []

    if row["ad_load"] > 0.8:
        reason.append("high ad load")

    if row["hour"] > 18:
        reason.append("peak hour usage")

    if row["disk_utilization"] > 90:
        reason.append("critical disk utilization")

    if row.get("is_injected_anomaly", False):
        reason.append("sudden DU spike pattern")

    return " + ".join(reason) if reason else "unusual feature combination"


def recommend_action(row: pd.Series) -> str:
    if row["disk_utilization"] > 100:
        return "Investigate device storage pressure and consider cleanup or cache purge."

    if row["ad_load"] > 0.8 and row["hour"] > 18:
        return "Check ad insertion workload during peak viewing window."

    if row["hour"] > 18:
        return "Monitor peak-hour DU trend for repeated spikes."

    return "Review recent device activity and compare against DMA/device-type baseline."


def classify_anomaly(row: pd.Series) -> str:
    if row["disk_utilization"] >= 90:
        return "high_du"

    if row["disk_utilization"] <= 35:
        return "unusually_low_du"

    return "unusual_context"


def build_insights(result: pd.DataFrame) -> pd.DataFrame:
    anomalies = result[result["anomaly"] == -1].copy()
    anomalies["anomaly_type"] = anomalies.apply(classify_anomaly, axis=1)
    anomalies["reason"] = anomalies.apply(explain, axis=1)
    anomalies["recommended_action"] = anomalies.apply(recommend_action, axis=1)

    columns = [
        "device_id",
        "hour",
        "ad_load",
        "disk_utilization",
        "anomaly_score",
        "anomaly_type",
        "is_injected_anomaly",
        "reason",
        "recommended_action",
    ]
    return anomalies[columns].sort_values(
        ["anomaly_type", "disk_utilization", "anomaly_score"],
        ascending=[True, False, True],
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_data = simulate_du_data()
    raw_data.to_csv(RAW_DATA_PATH, index=False)

    encoded_data, features = engineer_features(raw_data)
    result = detect_anomalies(encoded_data, features)
    insights = build_insights(result)
    insights.to_csv(ANOMALY_OUTPUT_PATH, index=False)

    print("DU data sample:")
    print(raw_data.head())
    print()

    print("Anomaly label counts (-1 = anomaly, 1 = normal):")
    print(result["anomaly"].value_counts().sort_index())
    print()

    print("Anomaly disk utilization summary:")
    print(insights["disk_utilization"].describe())
    print()

    print("Anomaly type counts:")
    print(insights["anomaly_type"].value_counts())
    print()

    print("Injected spike detection:")
    detected_injected = int(insights["is_injected_anomaly"].sum())
    print(f"Detected {detected_injected} of 100 injected spike rows.")
    print()

    print("Top anomaly insights:")
    print(insights.head(10).to_string(index=False))
    print()

    print(f"Saved raw data to: {RAW_DATA_PATH}")
    print(f"Saved anomaly insights to: {ANOMALY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
