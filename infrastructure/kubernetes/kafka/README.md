# FraudGuard Kafka (Async Pipeline)

This folder contains Kafka-related Kubernetes manifests.

## Purpose

Kafka is used to decouple:
- Real-time inference (FastAPI + Seldon)
- Slow, heavy tasks (SHAP, email alerts)

## Topics

### fraud-events
Emitted by FastAPI **only when fraud is detected**.

Consumed by:
- SHAP worker
- Alerting / Email worker

## Design Rules

- Kafka failures must NOT block inference
- Messages must be small and serializable
- Schema changes must be backward compatible
