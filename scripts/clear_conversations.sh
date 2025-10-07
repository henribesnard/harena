#!/bin/bash

# Script pour supprimer toutes les conversations via l'API
# Usage: ./clear_conversations.sh YOUR_JWT_TOKEN

TOKEN=$1

if [ -z "$TOKEN" ]; then
    echo "Usage: ./clear_conversations.sh YOUR_JWT_TOKEN"
    echo ""
    echo "Pour obtenir votre token, connectez-vous sur http://localhost:3000"
    echo "puis ouvrez la console du navigateur et tapez:"
    echo "  JSON.parse(localStorage.getItem('harena-auth')).state.token"
    exit 1
fi

echo "Suppression de toutes les conversations..."
curl -X DELETE "http://localhost:8000/api/v1/conversation/admin/conversations/clear" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"

echo ""
echo "Done!"
