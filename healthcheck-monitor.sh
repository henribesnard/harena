#!/bin/bash

# ============================================
# SCRIPT DE MONITORING CONTINU - HARENA
# ============================================
# Exécute des vérifications de santé périodiques
# À exécuter via cron : */5 * * * * /home/ec2-user/harena/healthcheck-monitor.sh

LOG_FILE="/var/log/harena-health.log"
ALERT_FILE="/tmp/harena-alerts.txt"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================
# Fonction de logging
# ============================================
log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

alert() {
    echo "[$TIMESTAMP] ❌ ALERTE: $1" | tee -a "$LOG_FILE" "$ALERT_FILE"
}

# ============================================
# VÉRIFICATIONS
# ============================================

log "=== Début du healthcheck ==="

# 1. Vérifier que tous les conteneurs sont UP
EXPECTED_CONTAINERS=(
    "harena_user_service_prod"
    "harena_search_service_prod"
    "harena_metric_service_prod"
    "harena_budget_profiling_prod"
    "harena_conversation_v3_prod"
    "harena-postgres"
    "harena-redis"
)

ISSUES_FOUND=false

for container in "${EXPECTED_CONTAINERS[@]}"; do
    STATUS=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo "not_found")

    if [ "$STATUS" != "running" ]; then
        alert "Container $container is $STATUS (expected: running)"
        ISSUES_FOUND=true
    else
        log "✅ $container: running"
    fi
done

# 2. Vérifier les healthchecks
UNHEALTHY=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | grep harena || true)
if [ ! -z "$UNHEALTHY" ]; then
    alert "Unhealthy containers detected: $UNHEALTHY"
    ISSUES_FOUND=true
else
    log "✅ All healthchecks passing"
fi

# 3. Vérifier les ports (pas de 127.0.0.1)
LOCALHOST_BINDINGS=$(docker ps --format "{{.Names}}: {{.Ports}}" | grep "127.0.0.1" | grep -E "user_service|search_service|metric_service|conversation" || true)
if [ ! -z "$LOCALHOST_BINDINGS" ]; then
    alert "Services bound to localhost only: $LOCALHOST_BINDINGS"
    ISSUES_FOUND=true
else
    log "✅ All services properly exposed on 0.0.0.0"
fi

# 4. Test des endpoints HTTP
test_endpoint() {
    local url=$1
    local service=$2
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" --max-time 5 || echo "000")

    if [ "$http_code" = "200" ]; then
        log "✅ $service endpoint: OK ($http_code)"
    else
        alert "$service endpoint failed: HTTP $http_code"
        ISSUES_FOUND=true
    fi
}

test_endpoint "http://localhost:3000/health" "user_service"
test_endpoint "http://localhost:3001/api/v1/search/health" "search_service"
test_endpoint "http://localhost:3002/health" "metric_service"
test_endpoint "http://localhost:3006/health" "budget_service"
test_endpoint "http://localhost:3008/health" "conversation_v3"

# 5. Vérifier l'utilisation des ressources
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

log "📊 Resources: Memory ${MEMORY_USAGE}%, Disk ${DISK_USAGE}%"

if [ "$MEMORY_USAGE" -gt 90 ]; then
    alert "High memory usage: ${MEMORY_USAGE}%"
    ISSUES_FOUND=true
fi

if [ "$DISK_USAGE" -gt 90 ]; then
    alert "High disk usage: ${DISK_USAGE}%"
    ISSUES_FOUND=true
fi

# 6. Vérifier les logs d'erreurs récentes (dernière heure)
ERROR_COUNT=$(docker-compose -f /home/ec2-user/harena/docker-compose.prod.yml logs --since 1h 2>&1 | grep -i "error\|exception\|failed" | wc -l)
log "📝 Errors in last hour: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 100 ]; then
    alert "High error rate: $ERROR_COUNT errors in last hour"
    ISSUES_FOUND=true
fi

# ============================================
# RÉSUMÉ
# ============================================

if [ "$ISSUES_FOUND" = true ]; then
    log "❌ Healthcheck FAILED - Issues detected"

    # Envoyer une alerte (optionnel: configurer email/slack)
    # mail -s "HARENA: Health issues detected" admin@example.com < "$ALERT_FILE"

    exit 1
else
    log "✅ Healthcheck PASSED - All systems operational"
    # Nettoyer les alertes
    > "$ALERT_FILE"
    exit 0
fi
