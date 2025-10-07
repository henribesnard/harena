"""
Script de validation de toutes les métriques
Compare les calculs du service métriques avec des requêtes SQL directes
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sqlalchemy import text
from db_service.session import get_db
from metric_service.core.calculator import metric_calculator
import json

# Configuration
USER_ID = 100  # User ID avec des transactions
OUTPUT_FILE = "metrics_validation_report.md"

class MetricsValidator:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = next(get_db())
        self.results = []

    def close(self):
        self.db.close()

    async def validate_mom_expenses(self):
        """Valide Month-over-Month pour les DÉPENSES"""
        print("Validating MoM Expenses...")

        # Calcul via service - Août 2025 - DÉPENSES
        service_result = await metric_calculator.calculate_mom(self.user_id, month="2025-08", transaction_type="expenses")

        # Calcul SQL direct - Août 2025
        query = text("""
            WITH current_month AS (
                SELECT
                    DATE_TRUNC('month', '2025-08-01'::date) as month,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= DATE_TRUNC('month', '2025-08-01'::date)
                  AND transaction_date < DATE_TRUNC('month', '2025-08-01'::date) + INTERVAL '1 month'
                  AND amount < 0
            ),
            previous_month AS (
                SELECT
                    DATE_TRUNC('month', '2025-07-01'::date) as month,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= DATE_TRUNC('month', '2025-07-01'::date)
                  AND transaction_date < DATE_TRUNC('month', '2025-08-01'::date)
                  AND amount < 0
            )
            SELECT
                cm.month as current_month,
                cm.total as current_amount,
                pm.month as previous_month,
                pm.total as previous_amount,
                (cm.total - pm.total) as change_amount,
                CASE
                    WHEN pm.total = 0 THEN 0
                    ELSE ((cm.total - pm.total) / ABS(pm.total) * 100)
                END as change_percent
            FROM current_month cm, previous_month pm
        """)

        sql_result = self.db.execute(query, {"user_id": self.user_id}).fetchone()

        self.results.append({
            "metric": "Month-over-Month (MoM) - DÉPENSES",
            "service": service_result if service_result else None,
            "sql": {
                "current_month": sql_result.current_month.isoformat() if sql_result else None,
                "current_amount": float(sql_result.current_amount) if sql_result else 0,
                "previous_month": sql_result.previous_month.isoformat() if sql_result else None,
                "previous_amount": float(sql_result.previous_amount) if sql_result else 0,
                "change_amount": float(sql_result.change_amount) if sql_result else 0,
                "change_percent": float(sql_result.change_percent) if sql_result else 0
            } if sql_result else None
        })

    async def validate_mom_income(self):
        """Valide Month-over-Month pour les REVENUS"""
        print("Validating MoM Income...")

        # Calcul via service - Août 2025 - REVENUS
        service_result = await metric_calculator.calculate_mom(self.user_id, month="2025-08", transaction_type="income")

        # Calcul SQL direct - Août 2025 - REVENUS
        query = text("""
            WITH current_month AS (
                SELECT
                    DATE_TRUNC('month', '2025-08-01'::date) as month,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= DATE_TRUNC('month', '2025-08-01'::date)
                  AND transaction_date < DATE_TRUNC('month', '2025-08-01'::date) + INTERVAL '1 month'
                  AND amount > 0
            ),
            previous_month AS (
                SELECT
                    DATE_TRUNC('month', '2025-07-01'::date) as month,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= DATE_TRUNC('month', '2025-07-01'::date)
                  AND transaction_date < DATE_TRUNC('month', '2025-08-01'::date)
                  AND amount > 0
            )
            SELECT
                cm.month as current_month,
                cm.total as current_amount,
                pm.month as previous_month,
                pm.total as previous_amount,
                (cm.total - pm.total) as change_amount,
                CASE
                    WHEN pm.total = 0 THEN 0
                    ELSE ((cm.total - pm.total) / ABS(pm.total) * 100)
                END as change_percent
            FROM current_month cm, previous_month pm
        """)

        sql_result = self.db.execute(query, {"user_id": self.user_id}).fetchone()

        self.results.append({
            "metric": "Month-over-Month (MoM) - REVENUS",
            "service": service_result if service_result else None,
            "sql": {
                "current_month": sql_result.current_month.isoformat() if sql_result else None,
                "current_amount": float(sql_result.current_amount) if sql_result else 0,
                "previous_month": sql_result.previous_month.isoformat() if sql_result else None,
                "previous_amount": float(sql_result.previous_amount) if sql_result else 0,
                "change_amount": float(sql_result.change_amount) if sql_result else 0,
                "change_percent": float(sql_result.change_percent) if sql_result else 0
            } if sql_result else None
        })

    async def validate_yoy_expenses(self):
        """Valide Year-over-Year pour les DÉPENSES"""
        print("Validating YoY Expenses...")

        # Calcul via service - DÉPENSES
        service_result = await metric_calculator.calculate_yoy(self.user_id, transaction_type="expenses")

        # Calcul SQL direct
        query = text("""
            WITH current_year AS (
                SELECT
                    EXTRACT(YEAR FROM CURRENT_DATE) as year,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = EXTRACT(YEAR FROM CURRENT_DATE)
                  AND amount < 0
            ),
            previous_year AS (
                SELECT
                    EXTRACT(YEAR FROM CURRENT_DATE) - 1 as year,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1
                  AND amount < 0
            )
            SELECT
                cy.year as current_year,
                cy.total as current_amount,
                py.year as previous_year,
                py.total as previous_amount,
                (cy.total - py.total) as change_amount,
                CASE
                    WHEN py.total = 0 THEN 0
                    ELSE ((cy.total - py.total) / ABS(py.total) * 100)
                END as change_percent
            FROM current_year cy, previous_year py
        """)

        sql_result = self.db.execute(query, {"user_id": self.user_id}).fetchone()

        self.results.append({
            "metric": "Year-over-Year (YoY) - DÉPENSES",
            "service": service_result if service_result else None,
            "sql": {
                "current_year": int(sql_result.current_year) if sql_result else None,
                "current_amount": float(sql_result.current_amount) if sql_result else 0,
                "previous_year": int(sql_result.previous_year) if sql_result else None,
                "previous_amount": float(sql_result.previous_amount) if sql_result else 0,
                "change_amount": float(sql_result.change_amount) if sql_result else 0,
                "change_percent": float(sql_result.change_percent) if sql_result else 0
            } if sql_result else None
        })

    async def validate_yoy_income(self):
        """Valide Year-over-Year pour les REVENUS"""
        print("Validating YoY Income...")

        # Calcul via service - REVENUS
        service_result = await metric_calculator.calculate_yoy(self.user_id, transaction_type="income")

        # Calcul SQL direct - REVENUS
        query = text("""
            WITH current_year AS (
                SELECT
                    EXTRACT(YEAR FROM CURRENT_DATE) as year,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = EXTRACT(YEAR FROM CURRENT_DATE)
                  AND amount > 0
            ),
            previous_year AS (
                SELECT
                    EXTRACT(YEAR FROM CURRENT_DATE) - 1 as year,
                    COALESCE(SUM(amount), 0) as total
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1
                  AND amount > 0
            )
            SELECT
                cy.year as current_year,
                cy.total as current_amount,
                py.year as previous_year,
                py.total as previous_amount,
                (cy.total - py.total) as change_amount,
                CASE
                    WHEN py.total = 0 THEN 0
                    ELSE ((cy.total - py.total) / ABS(py.total) * 100)
                END as change_percent
            FROM current_year cy, previous_year py
        """)

        sql_result = self.db.execute(query, {"user_id": self.user_id}).fetchone()

        self.results.append({
            "metric": "Year-over-Year (YoY) - REVENUS",
            "service": service_result if service_result else None,
            "sql": {
                "current_year": int(sql_result.current_year) if sql_result else None,
                "current_amount": float(sql_result.current_amount) if sql_result else 0,
                "previous_year": int(sql_result.previous_year) if sql_result else None,
                "previous_amount": float(sql_result.previous_amount) if sql_result else 0,
                "change_amount": float(sql_result.change_amount) if sql_result else 0,
                "change_percent": float(sql_result.change_percent) if sql_result else 0
            } if sql_result else None
        })

    async def validate_coverage_rate(self):
        """Valide le Taux de Couverture"""
        print("Validating Coverage Rate...")

        # Calcul via service - Août 2025
        service_result = await metric_calculator.calculate_coverage_rate(self.user_id, mois=8, annee=2025)

        # Calcul SQL direct - Août 2025
        query = text("""
            WITH month_transactions AS (
                SELECT
                    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_revenus,
                    SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_depenses
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND EXTRACT(YEAR FROM transaction_date) = 2025
                  AND EXTRACT(MONTH FROM transaction_date) = 8
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

        sql_result = self.db.execute(query, {"user_id": self.user_id}).fetchone()

        self.results.append({
            "metric": "Taux de Couverture (Coverage Rate)",
            "service": service_result if service_result else None,
            "sql": {
                "revenus": float(sql_result.revenus) if sql_result else 0,
                "depenses": float(sql_result.depenses) if sql_result else 0,
                "solde": float(sql_result.solde) if sql_result else 0,
                "taux_couverture": float(sql_result.taux_couverture) if sql_result else 0
            } if sql_result else None
        })

    async def validate_savings_rate(self):
        """Valide le taux d'épargne"""
        print("Validating Savings Rate...")

        period_days = 30

        # Calcul via service
        service_result = await metric_calculator.calculate_savings_rate(self.user_id, period_days)

        # Calcul SQL direct
        query = text("""
            WITH period_transactions AS (
                SELECT
                    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_income,
                    SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_expenses
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= CURRENT_DATE - INTERVAL ':period_days days'
                  AND transaction_date <= CURRENT_DATE
            )
            SELECT
                COALESCE(total_income, 0) as total_income,
                COALESCE(total_expenses, 0) as total_expenses,
                COALESCE(total_income, 0) - COALESCE(total_expenses, 0) as net_savings,
                CASE
                    WHEN COALESCE(total_income, 0) = 0 THEN 0
                    ELSE ((COALESCE(total_income, 0) - COALESCE(total_expenses, 0)) / total_income * 100)
                END as savings_rate
            FROM period_transactions
        """)

        sql_result = self.db.execute(query, {"user_id": self.user_id, "period_days": period_days}).fetchone()

        self.results.append({
            "metric": "Savings Rate",
            "service": service_result if service_result else None,
            "sql": {
                "total_income": float(sql_result.total_income) if sql_result else 0,
                "total_expenses": float(sql_result.total_expenses) if sql_result else 0,
                "net_savings": float(sql_result.net_savings) if sql_result else 0,
                "savings_rate": float(sql_result.savings_rate) if sql_result else 0
            } if sql_result else None
        })

    async def validate_burn_rate(self):
        """Valide le Burn Rate"""
        print("Validating Burn Rate...")

        period_days = 30

        # Calcul via service
        try:
            service_result = await metric_calculator.calculate_burn_rate(self.user_id, period_days)
        except Exception as e:
            print(f"  Warning: Service calculation failed: {e}")
            service_result = None

        # Calcul SQL direct - Simplifié sans la table accounts
        query = text("""
            WITH monthly_expenses AS (
                SELECT
                    AVG(ABS(amount)) as avg_daily_expense
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= CURRENT_DATE - :period_days * INTERVAL '1 day'
                  AND amount < 0
            )
            SELECT
                0.0 as current_balance,
                COALESCE(avg_daily_expense, 0) * 30 as monthly_burn_rate,
                NULL as runway_days
            FROM monthly_expenses
        """)

        try:
            sql_result = self.db.execute(query, {"user_id": self.user_id, "period_days": period_days}).fetchone()
        except Exception as e:
            print(f"  Warning: SQL calculation failed: {e}")
            sql_result = None

        self.results.append({
            "metric": "Burn Rate",
            "service": service_result if service_result else None,
            "sql": {
                "current_balance": float(sql_result.current_balance) if sql_result else 0,
                "monthly_burn_rate": float(sql_result.monthly_burn_rate) if sql_result else 0,
                "runway_days": float(sql_result.runway_days) if sql_result and sql_result.runway_days else None,
                "runway_months": float(sql_result.runway_days / 30) if sql_result and sql_result.runway_days else None
            } if sql_result else None
        })

    async def validate_recurring_expenses(self):
        """Valide les dépenses récurrentes"""
        print("Validating Recurring Expenses...")

        min_occurrences = 3
        lookback_days = 90

        # Calcul via service
        service_result = await metric_calculator.calculate_recurring_expenses(
            self.user_id, min_occurrences, lookback_days
        )

        # Calcul SQL direct (simplifié)
        query = text("""
            WITH merchant_expenses AS (
                SELECT
                    clean_description,
                    COUNT(*) as occurrences,
                    AVG(ABS(amount)) as average_amount,
                    SUM(ABS(amount)) as total_spent,
                    MIN(transaction_date) as first_occurrence,
                    MAX(transaction_date) as last_occurrence
                FROM raw_transactions
                WHERE user_id = :user_id
                  AND transaction_date >= CURRENT_DATE - INTERVAL ':lookback_days days'
                  AND amount < 0
                  AND clean_description IS NOT NULL
                GROUP BY clean_description
                HAVING COUNT(*) >= :min_occurrences
            )
            SELECT
                clean_description as merchant,
                occurrences,
                average_amount,
                total_spent,
                first_occurrence,
                last_occurrence
            FROM merchant_expenses
            ORDER BY total_spent DESC
            LIMIT 10
        """)

        sql_results = self.db.execute(query, {
            "user_id": self.user_id,
            "min_occurrences": min_occurrences,
            "lookback_days": lookback_days
        }).fetchall()

        self.results.append({
            "metric": "Recurring Expenses",
            "service": service_result if service_result else None,
            "sql": {
                "recurring_count": len(sql_results),
                "recurring_expenses": [
                    {
                        "merchant": row.merchant,
                        "occurrences": row.occurrences,
                        "average_amount": float(row.average_amount),
                        "total_spent": float(row.total_spent),
                        "first_occurrence": row.first_occurrence.isoformat(),
                        "last_occurrence": row.last_occurrence.isoformat()
                    }
                    for row in sql_results
                ] if sql_results else []
            }
        })

    def generate_markdown_report(self):
        """Génère un rapport Markdown"""
        print(f"\nGenerating report: {OUTPUT_FILE}...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# Rapport de Validation des Métriques\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**User ID:** {self.user_id}\n\n")
            f.write("---\n\n")

            for result in self.results:
                f.write(f"## {result['metric']}\n\n")

                # Service Result
                f.write("### 📊 Résultat Service Métriques\n\n")
                f.write("```json\n")
                f.write(json.dumps(result['service'], indent=2, default=str))
                f.write("\n```\n\n")

                # SQL Result
                f.write("### 🗄️ Résultat SQL Direct\n\n")
                f.write("```json\n")
                f.write(json.dumps(result['sql'], indent=2, default=str))
                f.write("\n```\n\n")

                # Comparison
                f.write("### ✅ Comparaison\n\n")
                self._compare_results(f, result)

                f.write("\n---\n\n")

    def _compare_results(self, f, result):
        """Compare les résultats et écrit les différences"""
        service = result.get('service')
        sql = result.get('sql')

        if not service or not sql:
            f.write("⚠️ **Données manquantes pour comparaison**\n\n")
            return

        # Comparer les champs numériques clés
        if "Month-over-Month" in result['metric']:
            self._compare_field(f, "Montant actuel", service.get('current_amount'), sql.get('current_amount'))
            self._compare_field(f, "Montant précédent", service.get('previous_amount'), sql.get('previous_amount'))
            self._compare_field(f, "Changement %", service.get('change_percent'), sql.get('change_percent'))

        elif "Year-over-Year" in result['metric']:
            self._compare_field(f, "Montant année actuelle", service.get('current_amount'), sql.get('current_amount'))
            self._compare_field(f, "Montant année précédente", service.get('previous_amount'), sql.get('previous_amount'))
            self._compare_field(f, "Changement %", service.get('change_percent'), sql.get('change_percent'))

        elif "Coverage Rate" in result['metric']:
            self._compare_field(f, "Revenus", service.get('revenus'), sql.get('revenus'))
            self._compare_field(f, "Dépenses", service.get('depenses'), sql.get('depenses'))
            self._compare_field(f, "Solde", service.get('solde'), sql.get('solde'))
            self._compare_field(f, "Taux de couverture %", service.get('taux_couverture'), sql.get('taux_couverture'))

        elif result['metric'] == "Savings Rate":
            self._compare_field(f, "Revenus totaux", service.get('total_income'), sql.get('total_income'))
            self._compare_field(f, "Dépenses totales", service.get('total_expenses'), sql.get('total_expenses'))
            self._compare_field(f, "Épargne nette", service.get('net_savings'), sql.get('net_savings'))
            self._compare_field(f, "Taux d'épargne %", service.get('savings_rate'), sql.get('savings_rate'))

        elif result['metric'] == "Burn Rate":
            self._compare_field(f, "Solde actuel", service.get('current_balance'), sql.get('current_balance'))
            self._compare_field(f, "Burn rate mensuel", service.get('monthly_burn_rate'), sql.get('monthly_burn_rate'))
            self._compare_field(f, "Runway (mois)", service.get('runway_months'), sql.get('runway_months'))

        elif result['metric'] == "Recurring Expenses":
            service_count = len(service.get('recurring_expenses', []))
            sql_count = sql.get('recurring_count', 0)
            f.write(f"- **Nombre de dépenses récurrentes:** Service={service_count}, SQL={sql_count}")
            if service_count == sql_count:
                f.write(" ✅\n")
            else:
                f.write(f" ⚠️ (Différence: {abs(service_count - sql_count)})\n")

    def _compare_field(self, f, field_name, service_value, sql_value, tolerance=0.01):
        """Compare un champ numérique avec tolérance"""
        f.write(f"- **{field_name}:** Service={service_value}, SQL={sql_value}")

        if service_value is None and sql_value is None:
            f.write(" ✅\n")
        elif service_value is None or sql_value is None:
            f.write(" ⚠️ (Valeur manquante)\n")
        else:
            try:
                service_float = float(service_value)
                sql_float = float(sql_value)
                diff = abs(service_float - sql_float)

                if diff <= tolerance:
                    f.write(" ✅\n")
                else:
                    f.write(f" ⚠️ (Différence: {diff:.2f})\n")
            except (ValueError, TypeError):
                if str(service_value) == str(sql_value):
                    f.write(" ✅\n")
                else:
                    f.write(" ⚠️ (Valeurs différentes)\n")

async def main():
    """Point d'entrée principal"""
    print("Starting Metrics Validation...\n")

    validator = MetricsValidator(USER_ID)

    try:
        # Valider les 5 métriques essentielles
        await validator.validate_mom_expenses()
        await validator.validate_mom_income()
        await validator.validate_yoy_expenses()
        await validator.validate_yoy_income()
        await validator.validate_coverage_rate()

        # Générer le rapport
        validator.generate_markdown_report()

        print(f"\nValidation complete! Report saved to: {OUTPUT_FILE}")

    finally:
        validator.close()

if __name__ == "__main__":
    asyncio.run(main())
