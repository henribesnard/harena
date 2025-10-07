"""
Script de validation du Taux de Couverture pour les 7 premiers mois de 2025
Compare les résultats API avec les requêtes SQL directes
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sqlalchemy import text
from db_service.session import get_db
from metric_service.core.calculator import metric_calculator
import json

# Configuration
USER_ID = 100
YEAR = 2025
MONTHS = range(1, 8)  # Janvier à Juillet 2025
OUTPUT_FILE = "coverage_validation_7months.md"

class CoverageValidator:
    def __init__(self, user_id: int, year: int):
        self.user_id = user_id
        self.year = year
        self.db = next(get_db())
        self.results = []

    def close(self):
        self.db.close()

    async def validate_month(self, mois: int):
        """Valide le taux de couverture pour un mois donné"""
        print(f"Validating coverage for {mois}/{self.year}...")

        # Calcul via API/Service
        service_result = await metric_calculator.calculate_coverage_rate(
            self.user_id,
            mois=mois,
            annee=self.year
        )

        # Calcul SQL direct
        query = text("""
            WITH month_transactions AS (
                SELECT
                    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_revenus,
                    SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_depenses
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = :year
                  AND EXTRACT(MONTH FROM transaction_date) = :month
            )
            SELECT
                COALESCE(total_revenus, 0) as revenus,
                COALESCE(total_depenses, 0) as depenses,
                COALESCE(total_revenus, 0) - COALESCE(total_depenses, 0) as solde,
                CASE
                    WHEN COALESCE(total_revenus, 0) = 0 THEN
                        CASE WHEN COALESCE(total_depenses, 0) > 0 THEN -100.0 ELSE 0.0 END
                    ELSE
                        ((COALESCE(total_revenus, 0) - COALESCE(total_depenses, 0)) / COALESCE(total_revenus, 1) * 100)
                END as taux_couverture
            FROM month_transactions
        """)

        sql_result = self.db.execute(
            query,
            {"user_id": self.user_id, "year": self.year, "month": mois}
        ).fetchone()

        # Stocker les résultats
        self.results.append({
            "mois": mois,
            "annee": self.year,
            "label": service_result.get("periode", {}).get("label", f"Mois {mois}"),
            "service": {
                "revenus": service_result.get("revenus", 0),
                "depenses": service_result.get("depenses", 0),
                "solde": service_result.get("solde", 0),
                "taux_couverture": service_result.get("taux_couverture", 0),
                "couleur": service_result.get("affichage", {}).get("couleur", ""),
                "niveau": service_result.get("affichage", {}).get("niveau", ""),
                "message": service_result.get("affichage", {}).get("message", "")
            },
            "sql": {
                "revenus": float(sql_result.revenus) if sql_result else 0,
                "depenses": float(sql_result.depenses) if sql_result else 0,
                "solde": float(sql_result.solde) if sql_result else 0,
                "taux_couverture": float(sql_result.taux_couverture) if sql_result else 0
            },
            "match": abs(service_result.get("taux_couverture", 0) - (float(sql_result.taux_couverture) if sql_result else 0)) < 0.01
        })

    def generate_markdown_report(self):
        """Génère un rapport Markdown avec les requêtes SQL"""
        print(f"\nGenerating report: {OUTPUT_FILE}...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# VALIDATION TAUX DE COUVERTURE - 7 PREMIERS MOIS 2025\n\n")
            f.write(f"**Date de validation:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**User ID:** {self.user_id}\n\n")
            f.write(f"**Année:** {self.year}\n\n")
            f.write("---\n\n")

            # Tableau récapitulatif
            f.write("## 📊 TABLEAU RÉCAPITULATIF\n\n")
            f.write("| Mois | Revenus | Dépenses | Solde | Taux Couverture | Niveau | Validation |\n")
            f.write("|------|---------|----------|-------|-----------------|--------|------------|\n")

            for result in self.results:
                service = result["service"]
                match_icon = "✅" if result["match"] else "❌"

                # Couleur emoji
                couleur_emoji = {
                    "green-dark": "🟢🟢",
                    "green": "🟢",
                    "green-light": "🟢⚪",
                    "orange": "🟠",
                    "red": "🔴",
                    "gray": "⚪"
                }.get(service["couleur"], "")

                f.write(f"| **{result['label']}** | ")
                f.write(f"{service['revenus']:.2f}€ | ")
                f.write(f"{service['depenses']:.2f}€ | ")
                f.write(f"{service['solde']:.2f}€ | ")
                f.write(f"**{service['taux_couverture']:.2f}%** {couleur_emoji} | ")
                f.write(f"{service['niveau']} | ")
                f.write(f"{match_icon} |\n")

            f.write("\n---\n\n")

            # Détails par mois
            f.write("## 📋 DÉTAILS PAR MOIS\n\n")

            for result in self.results:
                f.write(f"### {result['label']}\n\n")

                # Résultats API
                f.write("#### 🔵 Résultat API/Service\n\n")
                service = result["service"]
                f.write(f"- **Revenus:** {service['revenus']:.2f}€\n")
                f.write(f"- **Dépenses:** {service['depenses']:.2f}€\n")
                f.write(f"- **Solde:** {service['solde']:.2f}€\n")
                f.write(f"- **Taux de couverture:** {service['taux_couverture']:.2f}%\n")
                f.write(f"- **Niveau:** {service['niveau']}\n")
                f.write(f"- **Couleur:** {service['couleur']}\n")
                f.write(f"- **Message:** {service['message']}\n\n")

                # Résultats SQL
                f.write("#### 🗄️ Résultat SQL Direct\n\n")
                sql = result["sql"]
                f.write(f"- **Revenus:** {sql['revenus']:.2f}€\n")
                f.write(f"- **Dépenses:** {sql['depenses']:.2f}€\n")
                f.write(f"- **Solde:** {sql['solde']:.2f}€\n")
                f.write(f"- **Taux de couverture:** {sql['taux_couverture']:.2f}%\n\n")

                # Validation
                if result["match"]:
                    f.write("#### ✅ Validation\n\n")
                    f.write("**Les calculs correspondent parfaitement !**\n\n")
                else:
                    diff = abs(service['taux_couverture'] - sql['taux_couverture'])
                    f.write("#### ⚠️ Validation\n\n")
                    f.write(f"**Différence détectée:** {diff:.2f}%\n\n")

                # Requête SQL pour vérification manuelle
                f.write("#### 🔍 Requête SQL pour Vérification Manuelle\n\n")
                f.write("```sql\n")
                f.write(f"""-- Taux de Couverture {result['label']}
WITH month_transactions AS (
    SELECT
        SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_revenus,
        SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_depenses
    FROM raw_transactions
    WHERE user_id = {self.user_id}
      AND EXTRACT(YEAR FROM transaction_date) = {self.year}
      AND EXTRACT(MONTH FROM transaction_date) = {result['mois']}
)
SELECT
    COALESCE(total_revenus, 0) as revenus,
    COALESCE(total_depenses, 0) as depenses,
    COALESCE(total_revenus, 0) - COALESCE(total_depenses, 0) as solde,
    CASE
        WHEN COALESCE(total_revenus, 0) = 0 THEN
            CASE WHEN COALESCE(total_depenses, 0) > 0 THEN -100.0 ELSE 0.0 END
        ELSE
            ((COALESCE(total_revenus, 0) - COALESCE(total_depenses, 0)) / COALESCE(total_revenus, 1) * 100)
    END as taux_couverture
FROM month_transactions;
""")
                f.write("```\n\n")
                f.write("---\n\n")

            # Statistiques globales
            f.write("## 📈 STATISTIQUES GLOBALES\n\n")

            taux_list = [r["service"]["taux_couverture"] for r in self.results]
            revenus_total = sum(r["service"]["revenus"] for r in self.results)
            depenses_total = sum(r["service"]["depenses"] for r in self.results)
            solde_total = sum(r["service"]["solde"] for r in self.results)

            f.write(f"- **Taux moyen:** {sum(taux_list) / len(taux_list):.2f}%\n")
            f.write(f"- **Taux minimum:** {min(taux_list):.2f}%\n")
            f.write(f"- **Taux maximum:** {max(taux_list):.2f}%\n")
            f.write(f"- **Mois en déficit:** {sum(1 for t in taux_list if t < 0)}\n")
            f.write(f"- **Revenus totaux (7 mois):** {revenus_total:.2f}€\n")
            f.write(f"- **Dépenses totales (7 mois):** {depenses_total:.2f}€\n")
            f.write(f"- **Solde total (7 mois):** {solde_total:.2f}€\n\n")

            # Requête SQL pour statistiques globales
            f.write("### 🔍 Requête SQL pour Statistiques Globales\n\n")
            f.write("```sql\n")
            f.write(f"""-- Statistiques globales Janvier-Juillet {self.year}
SELECT
    EXTRACT(MONTH FROM transaction_date) as mois,
    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as revenus,
    SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as depenses,
    SUM(amount) as solde,
    CASE
        WHEN SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) = 0 THEN 0
        ELSE (SUM(amount) / SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) * 100)
    END as taux_couverture
FROM raw_transactions
WHERE user_id = {self.user_id}
  AND EXTRACT(YEAR FROM transaction_date) = {self.year}
  AND EXTRACT(MONTH FROM transaction_date) BETWEEN 1 AND 7
GROUP BY EXTRACT(MONTH FROM transaction_date)
ORDER BY mois;
""")
            f.write("```\n\n")

            f.write("---\n\n")
            f.write("**FIN DU RAPPORT**\n")


async def main():
    """Point d'entrée principal"""
    print("Starting Coverage Rate Validation for 7 months...\n")

    validator = CoverageValidator(USER_ID, YEAR)

    try:
        # Valider chaque mois
        for mois in MONTHS:
            await validator.validate_month(mois)

        # Générer le rapport
        validator.generate_markdown_report()

        print(f"\nValidation complete! Report saved to: {OUTPUT_FILE}")

    finally:
        validator.close()

if __name__ == "__main__":
    asyncio.run(main())
