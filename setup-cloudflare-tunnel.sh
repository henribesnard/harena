#!/bin/bash
set -e

echo "=== Installation de Cloudflare Tunnel ==="

# Télécharger cloudflared
echo "Téléchargement de cloudflared..."
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
chmod +x /tmp/cloudflared
sudo mv /tmp/cloudflared /usr/local/bin/cloudflared

# Vérifier l'installation
echo "Vérification de l'installation..."
cloudflared --version

echo ""
echo "=== Installation terminée ! ==="
echo ""
echo "Pour démarrer un tunnel temporaire (URL gratuite), exécutez:"
echo "  cloudflared tunnel --url http://localhost:80"
echo ""
echo "Vous recevrez une URL comme: https://harena-xyz.trycloudflare.com"
echo "Cette URL sera active tant que la commande tourne."
