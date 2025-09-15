#!/usr/bin/env bash
set -euo pipefail
TARGET=${1:-champion}
cat > traefik/dynamic.yml <<YAML
http:
routers:
rerank:
rule: PathPrefix(`/rerank`)
service: $TARGET
services:
champion:
loadBalancer:
servers:
- url: http://reranker_champion:8080
challenger:
loadBalancer:
servers:
- url: http://reranker_challenger:8080
YAML
echo "Blue/Green switched to $TARGET"