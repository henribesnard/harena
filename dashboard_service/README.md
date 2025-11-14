# Harena Dashboard Service

Service FastAPI fournissant des agrégations avancées pour le tableau de bord utilisateur Harena. Il complète `budget_profiling_service` en produisant des séries temporelles prêtes à afficher, des répartitions par catégories et des indicateurs synthétiques.

## Objectifs

- Centraliser les calculs nécessaires aux visualisations (courbes, camemberts, radars, etc.).
- Exposer une API légère (`/api/v1/dashboard/*`) consommée par le front Vite/React.
- Faciliter l'expérimentation (mock data par défaut) tout en restant prête à être branchée sur les vraies bases analytiques.

## Lancer le service

```bash
cd dashboard_service
python -m venv .venv && .venv/Scripts/activate  # ou source .venv/bin/activate sous Linux/macOS
pip install -r requirements.txt
uvicorn main:app --reload --port 3009
```

Le service expose alors les endpoints suivants :

- `GET /health` – vérification basique.
- `GET /api/v1/dashboard/overview` – KPI synthétiques + progression des objectifs.
- `GET /api/v1/dashboard/cash-flow` – série mensuelle revenus/dépenses/solde.
- `GET /api/v1/dashboard/category-breakdown` – répartition des dépenses.
- `GET /api/v1/dashboard/all` – payload combiné utilisé par le front.

Chaque endpoint retourne actuellement des données simulées. Le fichier `main.py` contient des TODO pour remplacer ces mocks par des requêtes vers vos data warehouses ou services métier.

## Structure

```
dashboard_service/
├── main.py
├── requirements.txt
└── README.md
```

## Intégration front

Le front consomme `GET /api/v1/dashboard/all`. Pensez à définir `VITE_DASHBOARD_SERVICE_URL` (ex : `http://localhost:3009`) côté front/infra pour que les appels passent correctement via `buildServiceURL('DASHBOARD', ...)`.

