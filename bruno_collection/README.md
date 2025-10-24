# 🧪 Collection Bruno - Harena AWS API

Collection prête à l'emploi pour tester les API AWS de Harena.

## 📦 Installation

1. **Installer Bruno** (si ce n'est pas déjà fait):
   - Télécharger depuis: https://www.usebruno.com/
   - Ou via chocolatey: `choco install bruno`

2. **Ouvrir la collection**:
   - Lancer Bruno
   - Click "Open Collection"
   - Sélectionner le dossier `bruno_collection`

## 🚀 Utilisation

### 1. Sélectionner l'environnement

En haut à droite dans Bruno, sélectionnez **"AWS Production"**

Variables disponibles:
- `base_url`: `http://63.35.52.216`
- `user_id`: `3` (modifiez selon votre utilisateur)
- `token`: Sera rempli automatiquement après login

### 2. Se connecter

1. Ouvrir `Auth` → `Login`
2. Modifier l'email et le password dans le body
3. Cliquer sur **Run**
4. Le token sera automatiquement sauvegardé dans l'environnement ✅

### 3. Tester les endpoints

Tous les autres endpoints utilisent automatiquement le `{{token}}`.

**Ordre recommandé**:
1. ✅ Health → Global Health (sans auth)
2. ✅ Auth → Login
3. ✅ Auth → Get User Info
4. ✅ Metrics → Dashboard
5. ✅ Conversation → Ask Question

## 📁 Structure

```
bruno_collection/
├── bruno.json                     # Config collection
├── environments/
│   └── AWS Production.bru         # Variables d'environnement
├── Auth/
│   ├── Login.bru                  # Login (sauvegarde auto du token)
│   └── Get User Info.bru          # Info utilisateur connecté
├── Metrics/
│   └── Dashboard.bru              # Métriques dashboard
├── Conversation/
│   └── Ask Question.bru           # Poser une question
└── Health/
    └── Global Health.bru          # Health check global
```

## 🔧 Ajouter de nouveaux endpoints

### Via Bruno UI:
1. Click droit sur un dossier → "New Request"
2. Configurez l'URL, méthode, body
3. Ajoutez `{{token}}` dans Auth → Bearer Token si nécessaire

### Via fichier .bru:
Créez un nouveau fichier `.bru` avec ce template:

```
meta {
  name: Mon Endpoint
  type: http
  seq: 1
}

get {
  url: {{base_url}}/api/v1/mon-endpoint
  body: none
  auth: bearer
}

auth:bearer {
  token: {{token}}
}
```

## 💡 Astuces

### Voir les variables
`Settings` → `Environment` → Voir toutes les variables

### Debugging
- Onglet `Response` → Voir headers, body, status
- Onglet `Console` → Voir les logs des scripts

### Scripts utiles

**Auto-save response data:**
```javascript
script:post-response {
  if (res.status === 200) {
    bru.setEnvVar("conversation_id", res.body.conversation_id);
  }
}
```

**Log pour debug:**
```javascript
script:post-response {
  console.log("Response:", res.body);
}
```

## 🐛 Dépannage

### Token expiré
Relancez `Auth → Login` pour obtenir un nouveau token.

### 504 Timeout
Les services AWS mettent du temps à répondre. Augmentez le timeout dans Bruno:
`Settings` → `Request` → `Timeout`: 60000ms

### CORS Error
Ça ne devrait pas arriver avec Bruno (pas de navigateur), mais si oui, vérifiez Nginx CORS config.

## 📚 Documentation Complète

Voir `AWS_API_ENDPOINTS.md` pour:
- Liste complète des endpoints
- Exemples de requêtes/réponses
- Notes sur l'authentification
- Timeouts Nginx

---

**Bon testing! 🚀**
