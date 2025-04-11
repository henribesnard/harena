# Guide des fonctionnalités de synchronisation bancaire
Voici l'explication détaillée de chaque fonctionnalité de synchronisation et comment un client peut les utiliser :

# 1. Connexion initiale d'un compte bancaire
Processus :

Le client se connecte à son compte Harena
Il accède à la section "Ajouter un compte bancaire"
L'application appelle l'endpoint /api/v1/users/bridge/connect-session pour créer une session Bridge Connect
Le client est redirigé vers l'URL Bridge Connect pour s'authentifier auprès de sa banque
Après authentification réussie, Bridge envoie un webhook item.created à votre serveur
Le système crée automatiquement les structures de suivi (SyncItem, SyncAccount)
Bridge commence à synchroniser les transactions et envoie des webhooks item.refreshed et item.account.updated

Endpoint utilisé : POST /api/v1/users/bridge/connect-session
# 2. Vérification de l'état de synchronisation
Processus :

Le client peut consulter l'état de ses connexions bancaires dans l'application
L'application appelle l'endpoint /api/v1/sync/status
Le système renvoie l'état actuel des items (connexions bancaires) :

Nombre total d'items et comptes
Items nécessitant une action utilisateur
Statut et codes d'erreur éventuels
Date de dernière synchronisation réussie



Endpoint utilisé : GET /api/v1/sync/status
# 3. Synchronisation manuelle (rafraîchissement)
Processus :

Le client peut demander une mise à jour manuelle de ses données bancaires
L'application appelle l'endpoint /api/v1/sync/refresh
Le système lance une synchronisation pour tous les items actifs de l'utilisateur
Les nouvelles transactions sont récupérées et stockées
L'application affiche les données mises à jour

Endpoint utilisé : POST /api/v1/sync/refresh
# 4. Reconnexion d'un compte en erreur
Processus :

Si un compte nécessite une action (identifiants expirés, authentification forte requise, etc.)
L'application affiche une notification au client avec l'action requise
Le client clique sur "Reconnecter" pour l'item concerné
L'application appelle l'endpoint /api/v1/sync/reconnect/{bridge_item_id}
Le système génère une URL de session Bridge Connect spécifique à cet item
Le client est redirigé vers l'URL pour effectuer l'action requise
Après résolution, les synchronisations reprennent automatiquement

Endpoint utilisé : POST /api/v1/sync/reconnect/{bridge_item_id}
# 5. Synchronisation automatique via webhooks
Processus :

Ce processus est entièrement automatique et ne nécessite aucune action du client
Bridge API envoie des webhooks à votre serveur quand :

Un item est créé ou mis à jour
Un compte est mis à jour avec de nouvelles transactions
Le statut d'un item change


Votre système traite ces événements et met à jour les données en conséquence
L'application affiche les données actualisées lors de la prochaine connexion du client

Endpoint utilisé : /webhooks/bridge (utilisé en interne, pas par le client)
# 6. Gestion des erreurs de connexion
Processus :

Quand une erreur se produit (identifiants invalides, OTP requis, etc.)
Bridge envoie un webhook avec le nouveau statut de l'item
Le système met à jour le statut et marque l'item comme nécessitant une action
L'application affiche une notification au client avec les instructions appropriées :

Pour statut 402 : "Vos identifiants bancaires sont incorrects"
Pour statut 1010 : "Authentification forte requise par votre banque"
Pour statut 429 : "Action requise sur le site de votre banque"



Information utilisée : Le statut et la description détaillée retournés par l'endpoint /api/v1/sync/status
Exemples concrets d'utilisation
Exemple 1 : Ajouter un nouveau compte bancaire
javascript// Côté frontend
async function connectBankAccount() {
  const response = await fetch('/api/v1/users/bridge/connect-session', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      callback_url: 'https://votreapp.com/callback',
      country_code: 'FR'
    })
  });
  
  const data = await response.json();
  // Rediriger vers l'URL de session Bridge
  window.location.href = data.connect_url;
}
Exemple 2 : Vérifier l'état des synchronisations
javascript// Côté frontend
async function checkSyncStatus() {
  const response = await fetch('/api/v1/sync/status', {
    headers: {
      'Authorization': `Bearer ${userToken}`
    }
  });
  
  const status = await response.json();
  
  // Afficher les informations de synchronisation
  if (status.needs_user_action) {
    // Afficher une alerte pour les items nécessitant une action
    status.items_needing_action.forEach(item => {
      showReconnectionAlert(item);
    });
  }
}
Exemple 3 : Reconnecter un compte en erreur
javascript// Côté frontend
async function reconnectItem(bridgeItemId) {
  const response = await fetch(`/api/v1/sync/reconnect/${bridgeItemId}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${userToken}`
    }
  });
  
  const data = await response.json();
  // Rediriger vers l'URL de reconnexion
  window.location.href = data.connect_url;
}
Ce guide devrait vous aider à comprendre comment chaque fonctionnalité est accessible et utilisable par les clients de votre application.