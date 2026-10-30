# Eir

Health, biometrics, and wearable telemetry service for the Yggdrasil homelab ecosystem.

Named after Eir, the Norse goddess of healing and medical care, this service serves as the central hub for passive health telemetry, wearable integrations, and morning biometrics.

## Capabilities

- **Withings Cloud Webhooks**: Morning weigh-ins, body composition (fat %, muscle mass, water %, bone mass), and subscription verification.
- **Apple Watch & Apple Health (via iOS Shortcuts)**: Daily recovery vitals (HRV, resting heart rate, sleep stages, active energy) and automated workout sync.
- **Time-Series Storage**: Canonical historical record of daily health and readiness metrics.
- **Ecosystem Integration**:
  - Exposes REST API and MCP endpoints for Hermes Agent and Antigravity.
  - Feeds morning biometrics into daily briefing and Obsidian Second Brain logs.
  - Supplies bodyweight context to Kratos for strength training calculations.
