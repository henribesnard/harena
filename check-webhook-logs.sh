#!/bin/bash
# Script pour suivre les logs des webhooks sur AWS

INSTANCE_ID="i-0011b978b7cea66dc"
REGION="eu-west-3"

echo "📋 Récupération des logs webhooks sur AWS..."
echo "=============================================="

# Commande pour suivre les logs en temps réel
COMMAND="docker logs -f harena_sync_service --tail 50 | grep -E '(webhook|bridge|POST /webhooks)'"

# Envoi de la commande via SSM
COMMAND_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$COMMAND\"]" \
    --region "$REGION" \
    --query 'Command.CommandId' \
    --output text)

echo "Command ID: $COMMAND_ID"
echo ""
echo "Attendez quelques secondes pour les résultats..."
sleep 5

# Récupération des logs
aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'StandardOutputContent' \
    --output text

echo ""
echo "=============================================="
echo "Pour voir tous les logs sync_service:"
echo "docker logs harena_sync_service --tail 100"
