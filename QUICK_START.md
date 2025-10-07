# ⚡ Quick Start - Déploiement AWS Harena

## 🚀 Déploiement en 5 étapes

### 1️⃣ Initialiser AWS (2 min)

```bash
chmod +x scripts/*.sh
./scripts/init-aws.sh
```

### 2️⃣ Créer l'infrastructure (15-20 min)

```bash
cd terraform
terraform apply
# Tapez "yes" pour confirmer
cd ..
```

### 3️⃣ Migrer les données (5 min)

```bash
# Installer PostgreSQL client si nécessaire
./scripts/migrate-data.sh
# Tapez "yes" pour confirmer
```

### 4️⃣ Déployer les applications (5 min)

```bash
# Backend
./scripts/deploy-backend.sh

# Frontend
./scripts/deploy-frontend.sh
```

### 5️⃣ Configurer GitHub Actions

1. Aller sur : `https://github.com/henribesnard/harena/settings/secrets/actions`
2. Ajouter les secrets :
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

✅ **C'est tout !** Chaque push sur `main` déploiera automatiquement.

---

## 📊 URLs importantes

Récupérer les URLs :
```bash
cd terraform
terraform output
```

Vous obtiendrez :
- **Backend API** : `http://[IP]:8000`
- **Frontend** : `https://[CLOUDFRONT].cloudfront.net`
- **SSH** : `aws ssm start-session --target [INSTANCE-ID]`

---

## 💰 Coûts

- **Mensuel** : ~14-20 EUR
- **Auto-shutdown actif** : 50% d'économie EC2
- **Budget AWS** : 20 EUR/mois avec alertes

---

## 🛠️ Commandes utiles

```bash
# Voir les logs
aws logs tail /aws/ec2/harena --follow

# Accéder à EC2
aws ssm start-session --target [INSTANCE-ID]

# Redéployer backend
./scripts/deploy-backend.sh

# Redéployer frontend
./scripts/deploy-frontend.sh

# Détruire tout (⚠️)
cd terraform && terraform destroy
```

---

## 📚 Documentation complète

Voir `DEPLOYMENT.md` pour :
- Configuration détaillée
- Dépannage
- Monitoring
- Maintenance

---

## ✅ Checklist post-déploiement

- [ ] Backend répond sur `/health`
- [ ] Frontend charge correctement
- [ ] Base de données migrée (vérifier les tables)
- [ ] GitHub Actions configuré
- [ ] Alertes email reçues (confirmer SNS)
- [ ] Auto-shutdown testé (vendredi 22h)

**Budget 6 mois** : 200 EUR ✅
