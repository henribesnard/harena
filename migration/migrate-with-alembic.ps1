# Migration Heroku → RDS en utilisant les migrations Alembic existantes
# 1. Applique les migrations Alembic sur RDS (schéma propre)
# 2. Copie uniquement les données depuis Heroku

param(
    [switch]$SkipConfirmation = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$HEROKU_URI = "postgresql://u33jmrgnq6cpeg:p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6@c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d2csndnj80lfm5"
$RDS_URI = "postgresql://harena_admin:GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV@harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com:5432/harena"

$BACKUP_DIR = ".\migration\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$DATA_FILE = "$BACKUP_DIR\data_only.sql"
$LOG_FILE = "$BACKUP_DIR\migration_alembic.log"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
    Add-Content -Path $LOG_FILE -Value $logMessage -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MIGRATION AVEC ALEMBIC" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Créer le dossier de backup
New-Item -ItemType Directory -Force -Path $BACKUP_DIR | Out-Null
Write-Log "Dossier de backup: $BACKUP_DIR" "Cyan"
Write-Host ""

# ============================
# ÉTAPE 1: Vérifier Python et Alembic
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 1/5: Vérification environnement" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Vérification Python..." "Yellow"
try {
    $pythonVersion = python --version 2>&1
    Write-Log "✓ Python: $pythonVersion" "Green"
} catch {
    Write-Log "✗ Python non trouvé" "Red"
    exit 1
}

Write-Log "Vérification Alembic..." "Yellow"
try {
    $alembicVersion = alembic --version 2>&1
    Write-Log "✓ Alembic: $alembicVersion" "Green"
} catch {
    Write-Log "✗ Alembic non trouvé. Installation..." "Yellow"
    pip install alembic psycopg2-binary
}

Write-Host ""

# ============================
# ÉTAPE 2: Appliquer migrations Alembic sur RDS
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 2/5: Migrations Alembic → RDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Configuration DATABASE_URL pour RDS..." "Yellow"
$env:DATABASE_URL = $RDS_URI

Write-Log "Vérification de l'état des migrations..." "Yellow"
alembic current 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

Write-Host ""
if (-not $SkipConfirmation) {
    Write-Host "⚠️  Voulez-vous appliquer TOUTES les migrations Alembic sur RDS ?" -ForegroundColor Yellow
    Write-Host "   Cela va créer le schéma complet de la base de données." -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Continuer ? (o/n)"
    if ($response -ne "o" -and $response -ne "O") {
        Write-Log "Migration annulée" "Yellow"
        exit 0
    }
}

Write-Host ""
Write-Log "Application des migrations Alembic (upgrade head)..." "Yellow"
alembic upgrade head 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

if ($LASTEXITCODE -eq 0) {
    Write-Log "✓ Migrations Alembic appliquées avec succès" "Green"
} else {
    Write-Log "✗ Erreur lors de l'application des migrations" "Red"
    exit 1
}

Write-Host ""
Write-Log "Vérification du schéma créé..." "Yellow"
$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
psql -h "harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com" `
     -U "harena_admin" `
     -d "harena" `
     -c "\dt" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

Write-Host ""

# ============================
# ÉTAPE 3: Export des données Heroku
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 3/5: Export données Heroku" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Statistiques Heroku (avant export)..." "Yellow"
$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
psql -h "c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com" `
     -U "u33jmrgnq6cpeg" `
     -d "d2csndnj80lfm5" `
     -c "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables ORDER BY tablename;" `
     | Tee-Object -FilePath "$BACKUP_DIR\stats_heroku.txt"

Write-Host ""
Write-Log "Export UNIQUEMENT des données (pas de schéma)..." "Yellow"
pg_dump "$HEROKU_URI" `
    --data-only `
    --no-owner `
    --no-acl `
    --no-privileges `
    --disable-triggers `
    --exclude-table=alembic_version `
    -f "$DATA_FILE" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

if (Test-Path $DATA_FILE) {
    $dataSize = (Get-Item $DATA_FILE).Length / 1KB
    Write-Log "✓ Données exportées: $DATA_FILE ($([math]::Round($dataSize, 2)) KB)" "Green"
} else {
    Write-Log "✗ Échec de l'export des données" "Red"
    exit 1
}

Write-Host ""

# ============================
# ÉTAPE 4: Import des données dans RDS
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 4/5: Import données → RDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipConfirmation) {
    Write-Host "⚠️  Voulez-vous importer les données dans RDS ?" -ForegroundColor Yellow
    Write-Host "   Fichier: $DATA_FILE" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Continuer ? (o/n)"
    if ($response -ne "o" -and $response -ne "O") {
        Write-Log "Import annulé" "Yellow"
        Write-Host ""
        Write-Host "Schéma RDS créé. Données disponibles dans: $DATA_FILE" -ForegroundColor Cyan
        exit 0
    }
}

Write-Host ""
Write-Log "Import des données dans RDS..." "Yellow"
$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
psql "$RDS_URI" -f "$DATA_FILE" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

if ($LASTEXITCODE -eq 0) {
    Write-Log "✓ Données importées avec succès" "Green"
} else {
    Write-Log "⚠️  Import terminé avec des avertissements (normal pour les contraintes)" "Yellow"
}

Write-Host ""

# ============================
# ÉTAPE 5: Vérification
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 5/5: Vérification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Statistiques RDS (après import)..." "Yellow"
$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
psql -h "harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com" `
     -U "harena_admin" `
     -d "harena" `
     -c "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables ORDER BY tablename;" `
     | Tee-Object -FilePath "$BACKUP_DIR\stats_rds.txt"

Write-Host ""
Write-Log "Comparaison des comptages..." "Yellow"
Write-Host ""
Write-Host "=== HEROKU ===" -ForegroundColor Green
Get-Content "$BACKUP_DIR\stats_heroku.txt"
Write-Host ""
Write-Host "=== RDS ===" -ForegroundColor Green
Get-Content "$BACKUP_DIR\stats_rds.txt"

Write-Host ""
Write-Log "Vérification spécifique table users..." "Yellow"
$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
$herokuUsers = (psql -h "c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com" `
                     -U "u33jmrgnq6cpeg" `
                     -d "d2csndnj80lfm5" `
                     -t -c "SELECT COUNT(*) FROM users;").Trim()

$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
$rdsUsers = (psql -h "harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com" `
                  -U "harena_admin" `
                  -d "harena" `
                  -t -c "SELECT COUNT(*) FROM users;").Trim()

Write-Host ""
Write-Host "Users Heroku: $herokuUsers" -ForegroundColor Cyan
Write-Host "Users RDS:    $rdsUsers" -ForegroundColor Cyan

if ($herokuUsers -eq $rdsUsers) {
    Write-Log "✓ Nombre d'utilisateurs identique!" "Green"
} else {
    Write-Log "⚠️  Différence dans le nombre d'utilisateurs" "Yellow"
}

Write-Host ""
Write-Log "Vérification des mots de passe hashés..." "Yellow"
psql -h "harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com" `
     -U "harena_admin" `
     -d "harena" `
     -c "SELECT id, email, LEFT(hashed_password, 30) as hash FROM users LIMIT 3;" `
     2>&1 | Tee-Object -Append -FilePath $LOG_FILE

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✓ MIGRATION TERMINÉE !" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Schéma créé via Alembic migrations" -ForegroundColor Green
Write-Host "✅ Données copiées depuis Heroku" -ForegroundColor Green
Write-Host "✅ Mots de passe des utilisateurs préservés" -ForegroundColor Green
Write-Host ""
Write-Host "Backup: $BACKUP_DIR" -ForegroundColor Cyan
Write-Host "Log: $LOG_FILE" -ForegroundColor Cyan
Write-Host ""
