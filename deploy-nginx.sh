#!/bin/bash
# Deploy nginx configuration

# Backup existing config
sudo cp /etc/nginx/conf.d/harena.conf /etc/nginx/conf.d/harena.conf.backup.$(date +%Y%m%d_%H%M%S)

# Copy nginx.conf from git repo
cd /home/ec2-user
sudo cp nginx.conf /etc/nginx/conf.d/harena.conf

# Test and reload
if sudo nginx -t; then
    echo "✅ Nginx configuration valid"
    sudo systemctl reload nginx
    echo "✅ Nginx reloaded successfully"
else
    echo "❌ Nginx configuration invalid"
    # Restore backup
    sudo cp /etc/nginx/conf.d/harena.conf.backup.* /etc/nginx/conf.d/harena.conf
    exit 1
fi
