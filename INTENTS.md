# Intent Reference

This document lists all intents recognized by Harena's MVP and their associated categories and suggested actions.

## 1. Transactions

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| TRANSACTION_SEARCH | FINANCIAL_QUERY | Rechercher toutes transactions sans filtre | ["list_transactions"] |
| SEARCH_BY_DATE | FINANCIAL_QUERY | Transactions pour une date/période | ["search_by_date","list_transactions"] |
| SEARCH_BY_AMOUNT | FINANCIAL_QUERY | Transactions par montant | ["filter_by_amount","list_transactions"] |
| SEARCH_BY_MERCHANT | FINANCIAL_QUERY | Transactions liées à un marchand précis | ["search_by_merchant","list_transactions"] |
| SEARCH_BY_CATEGORY | FINANCIAL_QUERY | Transactions par catégorie | ["search_by_category","list_transactions"] |
| SEARCH_BY_AMOUNT_AND_DATE | FINANCIAL_QUERY | Combinaison montant + date | ["filter_by_amount","search_by_date","list_transactions"] |
| SEARCH_BY_OPERATION_TYPE | FINANCIAL_QUERY | Transactions filtrées par type d’opération (débit, crédit, carte…) | ["filter_by_operation_type","list_transactions"] |
| SEARCH_BY_TEXT | FINANCIAL_QUERY | Recherche textuelle libre | ["search_by_text","list_transactions"] |
| COUNT_TRANSACTIONS | FINANCIAL_QUERY | Compter les transactions correspondant à une requête | ["count_transactions"] |
| MERCHANT_INQUIRY | FINANCIAL_QUERY | Analyse détaillée par marchand | ["search_by_merchant","merchant_breakdown"] |
| FILTER_REQUEST | FILTER_REQUEST | Raffiner une requête transactionnelle (ex. uniquement débits) | ["apply_filters"] |

## 2. Analyse de dépenses

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| SPENDING_ANALYSIS | SPENDING_ANALYSIS | Analyse globale des dépenses | ["calculate_total","spending_breakdown"] |
| SPENDING_ANALYSIS_BY_CATEGORY | SPENDING_ANALYSIS | Analyse par catégorie | ["calculate_total","spending_breakdown"] |
| SPENDING_ANALYSIS_BY_PERIOD | SPENDING_ANALYSIS | Analyse par période | ["calculate_total","time_breakdown"] |
| SPENDING_COMPARISON | SPENDING_ANALYSIS | Comparaison de périodes ou de catégories | ["compare_periods","compare_categories"] |
| TREND_ANALYSIS | TREND_ANALYSIS | Tendance/évolution des dépenses | ["trend_analysis","monthly_comparison"] |
| CATEGORY_ANALYSIS | SPENDING_ANALYSIS | Répartition par catégories | ["category_breakdown","spending_distribution"] |
| COMPARISON_QUERY | SPENDING_ANALYSIS | Comparaison ciblée (ex. restaurants vs courses) | ["compare_categories","budget_breakdown"] |

## 3. Soldes de comptes

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| BALANCE_INQUIRY | ACCOUNT_BALANCE | Solde général actuel | ["get_current_balance"] |
| ACCOUNT_BALANCE_SPECIFIC | ACCOUNT_BALANCE | Solde d’un compte précis | ["get_account_balance"] |
| BALANCE_EVOLUTION | ACCOUNT_BALANCE | Historique/évolution du solde | ["show_balance_trend"] |

## 4. Intentions conversationnelles

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| GREETING | GREETING | Bonjour, Salut… | ["greeting_response"] |
| CONFIRMATION | CONFIRMATION | Merci, parfait… | ["acknowledgment_response"] |
| CLARIFICATION | CLARIFICATION | Peux-tu préciser ? | ["clarification_request"] |
| GENERAL_QUESTION | GENERAL_QUESTION | Question générale ne correspondant à aucune autre intention | ["general_response"] |

## 5. Intentions non supportées

| Intent Type | Category | Example | Actions |
| --- | --- | --- | --- |
| TRANSFER_REQUEST | UNSUPPORTED | Faire un virement | [] |
| PAYMENT_REQUEST | UNSUPPORTED | Payer une facture | [] |
| CARD_BLOCK | UNSUPPORTED | Bloquer ma carte | [] |
| BUDGET_INQUIRY | UNSUPPORTED (future) | Où en est mon budget ? | [] |
| GOAL_TRACKING | UNSUPPORTED (future) | Objectifs d’épargne | [] |
| EXPORT_REQUEST | UNSUPPORTED (future) | Exporter transactions | [] |
| OUT_OF_SCOPE | UNSUPPORTED | Requête hors domaine (ex. recette de cuisine) | [] |

## 6. Intentions ambiguës ou erreurs

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| UNCLEAR_INTENT | UNCLEAR_INTENT | Intention ambiguë ou non reconnue | ["ask_to_rephrase"] |
| UNKNOWN | UNCLEAR_INTENT | Phrase inintelligible | ["ask_to_rephrase"] |
| TEST_INTENT | UNCLEAR_INTENT | Message de test («[TEST] ping») | ["no_action"] |
| ERROR | UNCLEAR_INTENT | Entrée corrompue | ["retry_or_contact_support"] |

