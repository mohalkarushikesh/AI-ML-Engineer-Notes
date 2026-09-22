# Prometheus Cheat Sheet

## What is Prometheus?

Prometheus is an open-source monitoring and alerting toolkit designed for collecting, storing, querying, and visualizing time-series metrics.

---

## Installation

### Docker

```bash
docker run -p 9090:9090 prom/prometheus
```

### Kubernetes

```bash
helm install prometheus prometheus-community/prometheus
```

---

## Key Components

- Prometheus Server
- Exporters
- Pushgateway
- Alertmanager
- PromQL
- Grafana (Visualization)

---

## Configuration Example

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: node
    static_configs:
      - targets:
        - localhost:9100
```

---

## Common Exporters

| Exporter | Purpose |
|-----------|---------|
| Node Exporter | Linux/Server Metrics |
| Blackbox Exporter | Endpoint Monitoring |
| MySQL Exporter | MySQL Monitoring |
| PostgreSQL Exporter | PostgreSQL Monitoring |
| Redis Exporter | Redis Monitoring |
| Kubernetes Metrics | Cluster Monitoring |

---

## Metric Types

### Counter

```text
http_requests_total
```

Only increases.

### Gauge

```text
memory_usage_bytes
```

Can increase or decrease.

### Histogram

```text
request_duration_seconds_bucket
```

Stores latency distribution.

### Summary

```text
request_duration_seconds
```

Stores quantiles.

---

## PromQL Basics

### Instant Vector

```promql
up
```

### Filter by Label

```promql
up{job="node"}
```

### Count Series

```promql
count(up)
```

### Sum

```promql
sum(http_requests_total)
```

### Average

```promql
avg(cpu_usage)
```

---

## Rate Queries

### Per-Second Rate

```promql
rate(http_requests_total[5m])
```

### Increase Over Time

```promql
increase(http_requests_total[1h])
```

---

## Aggregation

### Group By Job

```promql
sum(rate(http_requests_total[5m])) by (job)
```

### Top N

```promql
topk(5, cpu_usage)
```

---

## Node Exporter Metrics

### CPU Usage

```promql
100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

### Memory Usage

```promql
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
```

### Disk Usage

```promql
node_filesystem_avail_bytes
```

---

## Alert Rule Example

```yaml
groups:
- name: server-alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage > 80
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: CPU usage is high
```

---

## Alertmanager Route

```yaml
route:
  receiver: email

receivers:
- name: email
```

---

## Service Discovery

- Kubernetes
- Consul
- EC2
- Azure
- GCE
- Static Targets

---

## Useful PromQL Queries

### Target Health

```promql
up
```

### Failed Targets

```promql
up == 0
```

### HTTP Error Rate

```promql
rate(http_requests_total{status=~"5.."}[5m])
```

### Requests Per Second

```promql
sum(rate(http_requests_total[1m]))
```

### Container CPU

```promql
rate(container_cpu_usage_seconds_total[5m])
```

---

## Best Practices

1. Use labels wisely.
2. Avoid high-cardinality metrics.
3. Use Histograms for latency.
4. Keep scrape intervals reasonable.
5. Record common expensive queries.
6. Configure Alertmanager properly.
7. Visualize metrics in Grafana.
8. Monitor the Prometheus server itself.

---

## Architecture Flow

```text
Application
    ↓
Exporter / Metrics Endpoint
    ↓
Prometheus Server
    ↓
PromQL Queries
    ↓
Alertmanager
    ↓
Email / Slack / PagerDuty

or

Prometheus
    ↓
Grafana Dashboards
```
