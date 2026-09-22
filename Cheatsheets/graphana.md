# Grafana Cheat Sheet

## What is Grafana?

Grafana is an open-source observability and visualization platform used to create dashboards, monitor systems, analyze logs, and visualize metrics from multiple data sources.

---

## Installation

### Docker

```bash
docker run -d -p 3000:3000 --name grafana grafana/grafana
```

### Kubernetes

```bash
helm install grafana grafana/grafana
```

### Default Login

```text
Username: admin
Password: admin
```

---

## Core Components

- Dashboards
- Panels
- Data Sources
- Variables
- Alerts
- Explore
- Annotations
- Plugins

---

## Common Data Sources

- Prometheus
- Loki
- Elasticsearch
- InfluxDB
- PostgreSQL
- MySQL
- Azure Monitor
- CloudWatch

---

## Panel Types

- Time Series
- Stat
- Gauge
- Bar Chart
- Heatmap
- Table
- Pie Chart
- Logs
- Geomap

---

## Prometheus Queries

```promql
up
```

```promql
sum(rate(http_requests_total[1m]))
```

```promql
100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

---

## Loki Queries

```logql
{job="app"}
```

```logql
{job="app"} |= "ERROR"
```

---

## Alerting

Example condition:

```text
WHEN CPU > 80%
FOR 5m
```

Notification channels:
- Email
- Slack
- Microsoft Teams
- PagerDuty
- Webhook

---

## LGTM Stack

```text
L = Loki
G = Grafana
T = Tempo
M = Mimir
```

---

## Best Practices

1. Use dashboard variables.
2. Build reusable dashboards.
3. Set meaningful alert thresholds.
4. Correlate logs, metrics, and traces.
5. Use annotations for deployments.
6. Version control dashboard JSON.

---

## Architecture Flow

```text
Application
    ↓
Metrics / Logs / Traces
    ↓
Prometheus / Loki / Tempo
    ↓
Grafana Dashboard
    ↓
Alerting
    ↓
Email / Slack / Teams
```
