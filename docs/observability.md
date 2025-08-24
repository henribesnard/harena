# Observability

This service exposes Prometheus metrics and structured JSON logs to monitor enrichment operations.

## Metrics

Metrics are exported at `/metrics` and include:

- `enrichment_processing_seconds`: Histogram of single transaction processing time.
- `enrichment_batch_processing_seconds`: Histogram for batch processing time.
- `enrichment_cache_hits_total`: Counter of skipped transactions retrieved from cache.
- `enrichment_errors_total`: Counter of processing errors.
- `account_enrichment_seconds`: Histogram of end-to-end account enrichment time.
- `account_enrichment_cache_hits_total`: Cache hit counter at service level.
- `account_enrichment_errors_total`: Counter for account-level errors.
- `account_enrichment_data_quality_score`: Gauge representing the ratio of successful transactions in a batch.

## Logging

Logging is configured with `python-json-logger` to emit JSON records. Each transaction is tagged with a `correlation_id` combining the user and transaction identifiers, enabling end‑to‑end tracing across services.

## Dashboard

A sample Grafana dashboard is provided in [`docs/grafana_dashboard.json`](grafana_dashboard.json). It visualises:

1. Enrichment throughput
2. Error rates
3. Data‑quality score

Import the dashboard into Grafana and configure Prometheus as the data source to start monitoring.
