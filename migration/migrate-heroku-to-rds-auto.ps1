# Script automatique de migration Heroku → AWS RDS
# Utilise pg_dump et psql en ligne de commande

param(
    [switch]$SkipConfirmation = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$HEROKU_URI = "postgresql://u33jmrgnq6cpeg:p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6@c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d2csndnj80lfm5"
$RDS_URI = "postgresql://harena_admin:GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV@harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com:5432/harena"

$BACKUP_DIR = ".\migration\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$SCHEMA_FILE = "$BACKUP_DIR\schema.sql"
$DATA_FILE = "$BACKUP_DIR\data.sql"
$LOG_FILE = "$BACKUP_DIR\migration.log"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
    Add-Content -Path $LOG_FILE -Value $logMessage -ErrorAction SilentlyContinue
}

function Test-PostgreSQLTools {
    try {
        $pgDumpVersion = & pg_dump --version 2>&1
        $psqlVersion = & psql --version 2>&1
        return $true
    } catch {
        return $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MIGRATION HEROKU → AWS RDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier les outils PostgreSQL
Write-Log "Vérification des outils PostgreSQL..." "Yellow"
if (-not (Test-PostgreSQLTools)) {
    Write-Log "✗ pg_dump/psql non trouvés" "Red"
    Write-Host ""
    Write-Host "Installez PostgreSQL client avec:" -ForegroundColor Yellow
    Write-Host "  .\migration\install-postgresql.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Ou manuellement depuis:" -ForegroundColor Yellow
    Write-Host "  https://www.enterprisedb.com/downloads/postgres-postgresql-downloads" -ForegroundColor White
    exit 1
}

$pgVersion = & pg_dump --version
Write-Log "✓ PostgreSQL tools: $pgVersion" "Green"

# Créer le dossier de backup
New-Item -ItemType Directory -Force -Path $BACKUP_DIR | Out-Null
Write-Log "Dossier de backup: $BACKUP_DIR" "Cyan"
Write-Host ""

# ============================
# ÉTAPE 1: Test des connexions
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 1/6: Test des connexions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Test connexion Heroku..." "Yellow"
$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
try {
    $null = psql -h "c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com" `
                  -U "u33jmrgnq6cpeg" `
                  -d "d2csndnj80lfm5" `
                  -c "SELECT 1;" 2>&1
    Write-Log "✓ Connexion Heroku OK" "Green"
} catch {
    Write-Log "✗ Impossible de se connecter à Heroku" "Red"
    exit 1
}

Write-Log "Test connexion RDS..." "Yellow"
$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
try {
    $null = psql -h "harena-db-dev.cxkgo0wkw0wx.eu-west-1.rds.amazonaws.com" `
                  -U "harena_admin" `
                  -d "harena" `
                  -c "SELECT 1;" 2>&1
    Write-Log "✓ Connexion RDS OK" "Green"
} catch {
    Write-Log "✗ Impossible de se connecter à RDS" "Red"
    exit 1
}

Write-Host ""

# ============================
# ÉTAPE 2: Statistiques Heroku
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 2/6: Statistiques Heroku" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
Write-Log "Comptage des tables et lignes..." "Yellow"
psql -h "c7pvjrnjs0e7al.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com" `
     -U "u33jmrgnq6cpeg" `
     -d "d2csndnj80lfm5" `
     -c "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables ORDER BY tablename;" `
     | Tee-Object -FilePath "$BACKUP_DIR\stats_heroku.txt"

Write-Host ""

# ============================
# ÉTAPE 3: Export du schéma
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 3/6: Export du schéma" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Export du schéma Heroku..." "Yellow"
$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
pg_dump "$HEROKU_URI" --schema-only --no-owner --no-acl --no-privileges -f "$SCHEMA_FILE"

if (Test-Path $SCHEMA_FILE) {
    $schemaSize = (Get-Item $SCHEMA_FILE).Length / 1KB
    Write-Log "✓ Schéma exporté: $SCHEMA_FILE ($([math]::Round($schemaSize, 2)) KB)" "Green"
} else {
    Write-Log "✗ Échec de l'export du schéma" "Red"
    exit 1
}

Write-Host ""

# ============================
# ÉTAPE 4: Export des données
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 4/6: Export des données" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Export des données Heroku (peut prendre plusieurs minutes)..." "Yellow"
$env:PGPASSWORD = "p7c0d1fde10b21669bb99f2ddad71c6f9c689a56a5fd4853816e9726afe49dbc6"
pg_dump "$HEROKU_URI" --data-only --no-owner --no-acl --no-privileges --disable-triggers -f "$DATA_FILE"

if (Test-Path $DATA_FILE) {
    $dataSize = (Get-Item $DATA_FILE).Length / 1KB
    Write-Log "✓ Données exportées: $DATA_FILE ($([math]::Round($dataSize, 2)) KB)" "Green"
} else {
    Write-Log "✗ Échec de l'export des données" "Red"
    exit 1
}

Write-Host ""

# ============================
# ÉTAPE 5: Import dans RDS
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 5/6: Import dans RDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipConfirmation) {
    Write-Host "⚠️  Voulez-vous importer dans RDS ?" -ForegroundColor Yellow
    Write-Host "   Schéma: $SCHEMA_FILE" -ForegroundColor White
    Write-Host "   Données: $DATA_FILE" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Continuer ? (o/n)"
    if ($response -ne "o" -and $response -ne "O") {
        Write-Log "Import annulé par l'utilisateur" "Yellow"
        Write-Host ""
        Write-Host "Fichiers sauvegardés dans: $BACKUP_DIR" -ForegroundColor Cyan
        exit 0
    }
}

Write-Log "Import du schéma dans RDS..." "Yellow"
$env:PGPASSWORD = "GVTAB`$gFUU2RM^t9Gv6UlQ&8zDIrb`$vV"
psql "$RDS_URI" -f "$SCHEMA_FILE" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
Write-Log "✓ Schéma importé" "Green"

Write-Host ""
Write-Log "Import des données dans RDS (peut prendre plusieurs minutes)..." "Yellow"
psql "$RDS_URI" -f "$DATA_FILE" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
Write-Log "✓ Données importées" "Green"

Write-Host ""

# ============================
# ÉTAPE 6: Vérification
# ============================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ÉTAPE 6/6: Vérification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Log "Statistiques RDS après migration..." "Yellow"
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
Get-Content "$BACKUP_DIR\stats_heroku.txt" | Select-Object -Last 20
Write-Host ""
Write-Host "=== RDS ===" -ForegroundColor Green
Get-Content "$BACKUP_DIR\stats_rds.txt" | Select-Object -Last 20

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✓ MIGRATION TERMINÉE !" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup: $BACKUP_DIR" -ForegroundColor Cyan
Write-Host "Log: $LOG_FILE" -ForegroundColor Cyan
Write-Host ""
Write-Host "Les utilisateurs peuvent se connecter avec leurs mots de passe existants." -ForegroundColor Green
