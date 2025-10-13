# Script d'installation PostgreSQL client pour Windows
# Télécharge et installe uniquement les outils en ligne de commande

Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation PostgreSQL Client Tools" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Vérifier si PostgreSQL est déjà installé
$pgPath = "C:\Program Files\PostgreSQL\16\bin"
if (Test-Path "$pgPath\pg_dump.exe") {
    Write-Host "✓ PostgreSQL déjà installé dans: $pgPath" -ForegroundColor Green
    Write-Host ""
    & "$pgPath\pg_dump.exe" --version
    exit 0
}

Write-Host "Téléchargement de PostgreSQL 16..." -ForegroundColor Yellow
Write-Host ""
Write-Host "OPTION 1 (Recommandé) : Installer via winget"
Write-Host "  winget install PostgreSQL.PostgreSQL.16"
Write-Host ""
Write-Host "OPTION 2 : Télécharger manuellement"
Write-Host "  https://www.enterprisedb.com/downloads/postgres-postgresql-downloads"
Write-Host "  Version : PostgreSQL 16.x (Windows x86-64)"
Write-Host ""
Write-Host "Après installation, PostgreSQL sera dans:"
Write-Host "  C:\Program Files\PostgreSQL\16\bin"
Write-Host ""

$response = Read-Host "Installer avec winget maintenant ? (o/n)"
if ($response -eq "o" -or $response -eq "O") {
    Write-Host "Installation via winget..." -ForegroundColor Yellow
    winget install PostgreSQL.PostgreSQL.16

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Installation réussie!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Redémarrez PowerShell puis lancez:" -ForegroundColor Yellow
        Write-Host "  .\migration\migrate-heroku-to-rds-auto.ps1"
    } else {
        Write-Host ""
        Write-Host "✗ Erreur d'installation" -ForegroundColor Red
        Write-Host "Installez manuellement depuis:" -ForegroundColor Yellow
        Write-Host "  https://www.enterprisedb.com/downloads/postgres-postgresql-downloads"
    }
} else {
    Write-Host ""
    Write-Host "Installez manuellement puis relancez ce script." -ForegroundColor Yellow
}
