#!/usr/bin/env bash
set -euo pipefail

# Seed example transactions into Elasticsearch
python search_service/scripts/seed_transactions.py
