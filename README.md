# Intelligent DU Anomaly + Insight System

Detect abnormal disk utilization, explain likely causes, and recommend actions for a simulated DU streaming/device dataset.

## What This Builds

Pipeline:

```text
Simulated Data -> Feature Engineering -> Isolation Forest -> Anomaly Detection
                                                     -> Reasoning Layer
                                                     -> Insight / Action
```

Features generated:

- `device_id`
- `dma`
- `device_type`
- `hour`
- `content_type`
- `ad_load`
- `disk_utilization`

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/du_anomaly_insight.py
```

Outputs:

- `data/du_data.csv`: simulated DU dataset
- `outputs/du_anomaly_insights.csv`: detected anomalies with reason and recommended action

## Current ML Approach

- Numerical feature scaling with `StandardScaler`
- Categorical one-hot encoding with `pandas.get_dummies`
- Anomaly detection with `IsolationForest`
- Lightweight LLM-style reasoning layer using rules for:
  - high ad load
  - peak hour usage
  - critical DU
  - sudden DU spike pattern

## Next Upgrades

- Add model evaluation against injected anomaly labels.
- Add richer baselines by DMA, device type, and content type.
- Store anomaly explanations as embeddings for similarity search.
- Replace simulated reasoning with an LLM or Bedrock Agent.
- Add a simple dashboard for trend views and anomaly drilldown.
