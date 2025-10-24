"""
Service pour récupérer et formatter le profil budgétaire utilisateur
depuis le budget_profiling_service
"""
import httpx
import logging
from typing import Optional, Dict, Any
from ..config.settings import settings

logger = logging.getLogger(__name__)


class UserProfileService:
    """
    Service pour interagir avec budget_profiling_service
    et récupérer le profil budgétaire de l'utilisateur
    """

    def __init__(self, budget_service_url: Optional[str] = None):
        """
        Args:
            budget_service_url: URL du budget_profiling_service
                               (default: from settings)
        """
        self.budget_service_url = budget_service_url or getattr(settings, 'BUDGET_SERVICE_URL', 'http://localhost:3006')
        self.http_client = httpx.AsyncClient(
            timeout=5.0,  # Timeout court pour éviter de bloquer
            follow_redirects=True
        )
        logger.info(f"UserProfileService initialized with URL: {self.budget_service_url}")

    async def get_user_profile(
        self,
        user_id: int,
        jwt_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère le profil budgétaire de l'utilisateur depuis budget_profiling_service

        Args:
            user_id: ID de l'utilisateur
            jwt_token: Token JWT pour l'authentification (optionnel)

        Returns:
            Dict contenant le profil budgétaire ou None si non disponible
        """
        try:
            headers = {}
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"

            url = f"{self.budget_service_url}/api/v1/budget/profile"
            logger.debug(f"Fetching user profile from {url} for user_id={user_id}")

            response = await self.http_client.get(
                url,
                headers=headers
            )

            if response.status_code == 200:
                profile = response.json()
                logger.info(
                    f"User profile loaded: user_id={user_id}, "
                    f"segment={profile.get('user_segment')}, "
                    f"completeness={profile.get('profile_completeness', 0):.0%}"
                )
                return profile

            elif response.status_code == 404:
                logger.info(f"User profile not found for user_id={user_id} (not analyzed yet)")
                return None

            else:
                logger.warning(
                    f"Failed to fetch user profile: status={response.status_code}, "
                    f"response={response.text[:200]}"
                )
                return None

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching user profile for user_id={user_id}")
            return None

        except Exception as e:
            logger.error(f"Error fetching user profile for user_id={user_id}: {e}")
            return None

    def format_profile_for_prompt(self, profile: Optional[Dict[str, Any]]) -> str:
        """
        Formate le profil utilisateur pour l'injection dans le prompt LLM

        Args:
            profile: Profil budgétaire (peut être None)

        Returns:
            Texte formaté pour le prompt
        """
        if not profile:
            return """**PROFIL UTILISATEUR :**
Profil budgétaire non disponible. L'utilisateur n'a pas encore été analysé.
→ Reste neutre dans tes réponses, sans faire de suppositions sur sa situation financière.
"""

        # Extraire les données du profil
        segment = profile.get('user_segment', 'indéterminé')
        pattern = profile.get('behavioral_pattern', 'indéterminé')
        income = profile.get('avg_monthly_income', 0)
        expenses = profile.get('avg_monthly_expenses', 0)
        savings = profile.get('avg_monthly_savings', 0)
        savings_rate = profile.get('savings_rate', 0)
        fixed = profile.get('fixed_charges_total', 0)
        semi_fixed = profile.get('semi_fixed_charges_total', 0)
        variable = profile.get('variable_charges_total', 0)
        remaining = profile.get('remaining_to_live', 0)
        completeness = profile.get('profile_completeness', 0)

        # Nouvelles métriques avancées
        health_score = profile.get('financial_health_score')
        risk_level = profile.get('risk_level', 'unknown')
        alerts = profile.get('active_alerts', [])

        # Formatter pour le prompt
        profile_text = f"""**PROFIL UTILISATEUR :**

📊 **Segmentation :**
- Segment budgétaire : {segment}
- Pattern comportemental : {pattern}
- Complétude du profil : {completeness:.0%}"""

        if health_score is not None:
            profile_text += f"\n- Score de santé financière : {health_score:.0f}/100"
        if risk_level and risk_level != 'unknown':
            profile_text += f"\n- Niveau de risque : {risk_level.upper()}"

        profile_text += f"""

💰 **Métriques mensuelles moyennes :**
- Revenus : {income:.2f}€
- Dépenses : {expenses:.2f}€
- Épargne : {savings:.2f}€ ({savings_rate:.1f}%)

📈 **Répartition des charges :**
- Charges fixes : {fixed:.2f}€
- Charges semi-fixes : {semi_fixed:.2f}€
- Charges variables : {variable:.2f}€
- Reste à vivre : {remaining:.2f}€"""

        if alerts and len(alerts) > 0:
            profile_text += f"\n\n🚨 **Alertes actives ({len(alerts)}) :**"
            for alert in alerts[:3]:  # Limiter à 3 alertes
                profile_text += f"\n- [{alert.get('level', 'INFO')}] {alert.get('title', 'Alerte')}"

        profile_text += f"""

💡 **Instructions de personnalisation :**
{self._get_personalization_guidelines(segment, savings_rate, pattern, risk_level)}
"""

        return profile_text

    def _get_personalization_guidelines(
        self,
        segment: str,
        savings_rate: float,
        pattern: str,
        risk_level: str
    ) -> str:
        """
        Génère des guidelines de personnalisation selon le profil

        Args:
            segment: Segment budgétaire (budget_serré, équilibré, confortable)
            savings_rate: Taux d'épargne en pourcentage
            pattern: Pattern comportemental
            risk_level: Niveau de risque

        Returns:
            Guidelines textuelles pour le LLM
        """
        guidelines = []

        # Guidelines selon le segment et le taux d'épargne
        if segment == 'budget_serré' or savings_rate < 10:
            guidelines.append(
                "→ Utilisateur en situation budgétaire TENDUE\n"
                "→ Privilégier les conseils d'OPTIMISATION et de RÉDUCTION des dépenses\n"
                "→ Être ENCOURAGEANT et proposer des petits objectifs réalisables\n"
                "→ ALERTER sur les dépenses inhabituelles ou importantes"
            )
        elif segment == 'confortable' and savings_rate > 30:
            guidelines.append(
                "→ Utilisateur avec situation budgétaire CONFORTABLE\n"
                "→ Proposer des stratégies d'ÉPARGNE et d'INVESTISSEMENT\n"
                "→ Mettre en avant les opportunités d'optimisation fiscale\n"
                "→ Être plus AMBITIEUX dans les recommandations"
            )
        else:
            guidelines.append(
                "→ Utilisateur avec situation budgétaire ÉQUILIBRÉE\n"
                "→ Équilibrer entre optimisation et qualité de vie\n"
                "→ Proposer des marges de progression RAISONNABLES\n"
                "→ Encourager le maintien de bonnes habitudes"
            )

        # Guidelines selon le niveau de risque
        if risk_level == 'high':
            guidelines.append(
                "→ RISQUE ÉLEVÉ détecté : Alerter l'utilisateur avec tact sur sa situation"
            )
        elif risk_level == 'medium':
            guidelines.append(
                "→ Risque modéré : Suggérer des améliorations préventives"
            )

        # Guidelines selon le pattern comportemental
        if pattern == 'acheteur_impulsif' or pattern == 'erratic_spender':
            guidelines.append(
                "→ Pattern impulsif/erratique détecté : suggérer de grouper les achats, planifier les dépenses"
            )
        elif pattern == 'planificateur':
            guidelines.append(
                "→ Pattern planificateur : valoriser sa constance, proposer des optimisations fines"
            )
        elif pattern in ['dépensier_hebdomadaire', 'high_frequency_spender']:
            guidelines.append(
                "→ Pattern haute fréquence : adapter les recommandations à ce rythme de dépenses"
            )

        return "\n".join(guidelines)

    async def close(self):
        """Ferme le client HTTP"""
        await self.http_client.aclose()
