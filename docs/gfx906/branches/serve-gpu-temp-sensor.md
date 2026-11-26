# `serve/gpu-temp-sensor`

Lets the Monitor show the AMD card's junction temperature instead of the edge. Own branch on upstream.

## What it does

amdgpu exposes up to three hwmon temperatures, told apart by `tempN_label`: edge, junction (the hottest spot on the die,
what the card throttles on) and memory. The Monitor, `/metrics` and Prometheus read `temp1` (edge), which on an MI50
under load sits 20–25 °C below the junction (62 / 81 °C decoding, 68 / 93 °C in a 177K-token prefill).

## Setting

| Config key | Values | Production |
| --- | --- | --- |
| `"gpu_temp_sensor"` | `edge` (default), `junction`, `mem` | `junction` |

A card without the chosen probe falls back to edge; NVIDIA and Metal ignore the key; any other value stops the server
at start with a message. It is editable in the Settings view. Tests: `serve.test_server` (`test_temp_sensor`).
