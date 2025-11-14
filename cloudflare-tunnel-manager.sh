#!/bin/bash
# Cloudflare Tunnel Manager pour Harena
# Gestion du tunnel Cloudflare sur EC2

set -e

INSTANCE_ID="i-0011b978b7cea66dc"
REGION="eu-west-1"

function send_command() {
    local commands="$1"
    local description="$2"

    echo "⏳ $description..."
    COMMAND_ID=$(aws ssm send-command \
        --instance-ids $INSTANCE_ID \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$commands\"]" \
        --region $REGION \
        --output text --query 'Command.CommandId')

    sleep 3

    aws ssm get-command-invocation \
        --command-id $COMMAND_ID \
        --instance-id $INSTANCE_ID \
        --region $REGION \
        --query 'StandardOutputContent' \
        --output text
}

function show_status() {
    echo "📊 Statut du Cloudflare Tunnel"
    echo "================================"
    send_command "ps aux | grep cloudflared | grep -v grep || echo 'Tunnel non actif'" "Vérification du processus"
}

function show_url() {
    echo "🔗 URL Cloudflare actuelle"
    echo "================================"
    send_command "cat /home/ec2-user/cloudflare-tunnel.log 2>/dev/null | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || echo 'Aucune URL trouvée. Le tunnel est peut-être arrêté.'" "Récupération de l'URL"
}

function start_tunnel() {
    echo "🚀 Démarrage du tunnel Cloudflare"
    echo "================================"
    send_command "pkill cloudflared 2>/dev/null || true; nohup cloudflared tunnel --url http://localhost:80 > /home/ec2-user/cloudflare-tunnel.log 2>&1 & sleep 5; cat /home/ec2-user/cloudflare-tunnel.log | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1" "Démarrage du tunnel"
    echo ""
    echo "✅ Tunnel démarré ! Utilisez l'URL ci-dessus pour accéder à Harena."
}

function stop_tunnel() {
    echo "⛔ Arrêt du tunnel Cloudflare"
    echo "================================"
    send_command "pkill cloudflared && echo 'Tunnel arrêté' || echo 'Tunnel déjà arrêté'" "Arrêt du tunnel"
}

function restart_tunnel() {
    echo "🔄 Redémarrage du tunnel Cloudflare"
    echo "================================"
    stop_tunnel
    echo ""
    sleep 2
    start_tunnel
}

function show_logs() {
    echo "📋 Logs du tunnel Cloudflare"
    echo "================================"
    send_command "tail -30 /home/ec2-user/cloudflare-tunnel.log 2>/dev/null || echo 'Pas de logs disponibles'" "Récupération des logs"
}

# Menu principal
case "${1:-help}" in
    status)
        show_status
        echo ""
        show_url
        ;;
    url)
        show_url
        ;;
    start)
        start_tunnel
        ;;
    stop)
        stop_tunnel
        ;;
    restart)
        restart_tunnel
        ;;
    logs)
        show_logs
        ;;
    help|*)
        echo "Cloudflare Tunnel Manager - Harena"
        echo "==================================="
        echo ""
        echo "Usage: $0 {status|url|start|stop|restart|logs}"
        echo ""
        echo "Commandes:"
        echo "  status   - Afficher le statut du tunnel et l'URL"
        echo "  url      - Afficher uniquement l'URL actuelle"
        echo "  start    - Démarrer le tunnel (génère une nouvelle URL)"
        echo "  stop     - Arrêter le tunnel"
        echo "  restart  - Redémarrer le tunnel (génère une nouvelle URL)"
        echo "  logs     - Afficher les derniers logs du tunnel"
        echo ""
        echo "Exemples:"
        echo "  $0 status    # Vérifier si le tunnel est actif"
        echo "  $0 url       # Obtenir l'URL Cloudflare actuelle"
        echo "  $0 restart   # Redémarrer et obtenir une nouvelle URL"
        ;;
esac
