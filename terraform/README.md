# Harena Infrastructure - Terraform

Infrastructure optimisée pour Harena Finance Platform avec budget de 20$/mois.

## Architecture

**All-in-One EC2 Instance** (t4g.small - ARM64)
- PostgreSQL 16 (port 5432)
- Redis 7 (port 6379)
- Elasticsearch 8 (port 9200)
- Backend API (port 8000)
- Frontend (port 80/443)

**Coût estimé**: ~19.50$/mois

## Prérequis

1. Terraform >= 1.5.0
2. AWS CLI configuré avec vos credentials
3. Votre adresse IP publique

## Configuration

1. **Éditer terraform.tfvars**
   ```bash
   cd terraform
   # Vérifier/modifier les valeurs, notamment allowed_ip
   ```

2. **Obtenir votre IP actuelle**
   ```bash
   curl https://api.ipify.org
   # Mettre à jour allowed_ip dans terraform.tfvars avec: x.x.x.x/32
   ```

## Déploiement

```bash
# 1. Initialiser Terraform
terraform init

# 2. Voir le plan de déploiement
terraform plan

# 3. Déployer l'infrastructure
terraform apply

# 4. Noter les outputs (IP, endpoints, etc.)
terraform output
```

## Connexion aux Services

### PostgreSQL (DBeaver)
```
Host: <elastic_ip_from_output>
Port: 5432
Database: harena
Username: harena_admin
Password: HaReNa2024SecureDbPassword123
```

### Redis
```
Host: <elastic_ip_from_output>
Port: 6379
Auth: HaReNa2024-Redis-Auth-Token-Secure-Key-123456
```

### Elasticsearch
```
URL: http://<elastic_ip_from_output>:9200
```

### SSH (via AWS SSM)
```bash
aws ssm start-session --target <instance_id_from_output> --region eu-west-1
```

## Vérifier les Services

```bash
# Se connecter à l'instance
aws ssm start-session --target <instance_id> --region eu-west-1

# Vérifier Docker Compose
cd /opt/harena
sudo docker-compose ps

# Voir les logs
sudo docker-compose logs -f
```

## Mise à jour de votre IP

Si votre IP change:

```bash
# Éditer terraform.tfvars avec la nouvelle IP
# Puis appliquer les changements
terraform apply
```

## Destruction

⚠️ **ATTENTION**: Cela détruira toutes les ressources et données !

```bash
terraform destroy
```

## Coûts

| Ressource | Configuration | Coût/mois |
|-----------|--------------|-----------|
| EC2 t4g.small | 2 vCPU, 2GB RAM | ~14.60$ |
| EBS gp3 30GB | Stockage | ~2.40$ |
| Elastic IP | Attaché | 0$ |
| Transfert | ~10GB | ~1.00$ |
| Snapshots | Optionnel | ~1.50$ |
| **TOTAL** | | **~19.50$** |

## Sécurité

- PostgreSQL accessible uniquement depuis votre IP
- Redis accessible uniquement depuis votre IP
- Elasticsearch accessible uniquement depuis votre IP
- Backend/Frontend accessibles publiquement
- SSH uniquement via AWS Systems Manager (pas de port 22 ouvert)
- Volumes EBS chiffrés
- IMDSv2 activé sur EC2
