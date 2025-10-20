"""Module 2: SQL Generation using DeepSeek API."""

from openai import AsyncOpenAI
from .intent_analyzer import IntentAnalysis
import asyncpg
import os
from typing import Optional

# DeepSeek client
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key"),
    base_url="https://api.deepseek.com"
)


async def get_database_schema() -> str:
    """
    Retrieve the actual database schema from PostgreSQL.

    Returns:
        str: Database schema description with tables and columns
    """
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            database=os.getenv("POSTGRES_DB", "harena"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )

        # Query to get table schema for raw_transactions and related tables
        schema_query = """
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            pg_catalog.col_description(
                (SELECT oid FROM pg_class WHERE relname = c.table_name AND relnamespace =
                    (SELECT oid FROM pg_namespace WHERE nspname = 'public')),
                c.ordinal_position
            ) as column_comment
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
            AND c.table_name IN ('raw_transactions', 'users', 'accounts', 'categories')
        ORDER BY c.table_name, c.ordinal_position;
        """

        rows = await conn.fetch(schema_query)

        # Build schema description
        schema_dict = {}
        for row in rows:
            table_name = row['table_name']
            if table_name not in schema_dict:
                schema_dict[table_name] = []

            col_desc = f"  {row['column_name']} {row['data_type'].upper()}"
            if row['is_nullable'] == 'NO':
                col_desc += " NOT NULL"
            if row['column_default']:
                col_desc += f" DEFAULT {row['column_default']}"
            if row['column_comment']:
                col_desc += f"  -- {row['column_comment']}"

            schema_dict[table_name].append(col_desc)

        # Format as text
        schema_text = ""
        for table_name, columns in schema_dict.items():
            schema_text += f"TABLE {table_name} (\n"
            schema_text += ",\n".join(columns)
            schema_text += "\n);\n\n"

        await conn.close()

        return schema_text.strip()

    except Exception as e:
        print(f"Error fetching database schema: {e}")
        # Fallback to basic schema
        return """
TABLE raw_transactions (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  amount DECIMAL(10,2) NOT NULL,
  transaction_date TIMESTAMP NOT NULL,
  category VARCHAR(100),
  subcategory VARCHAR(100),
  merchant VARCHAR(255),
  payment_method VARCHAR(50),
  description TEXT,
  is_recurring BOOLEAN
);
"""

SQL_GENERATION_SYSTEM_PROMPT = """
Tu es un expert PostgreSQL spécialisé en génération de requêtes SQL pour l'analyse financière.

RÈGLES STRICTES:
1. UNIQUEMENT SELECT (jamais INSERT/UPDATE/DELETE/DROP)
2. TOUJOURS filtrer par user_id = :user_id (OBLIGATOIRE - SÉCURITÉ)
3. Utiliser des paramètres préparés (:param_name)
4. La requête DOIT retourner 3 sections dans le JSON:
   a) "search_summary": résumé de recherche (total_results, amount_total, amount_avg, etc.)
   b) "aggregations": agrégations structurées (by_period, by_category, by_merchant, by_payment)
   c) "top_50_transactions": les 50 premières transactions détaillées
5. Utiliser des CTEs (WITH) pour la clarté
6. Gérer les NULL avec COALESCE
7. Utiliser DECIMAL pour les montants

SCHÉMA BASE DE DONNÉES:
{db_schema}

STRUCTURE DE REQUÊTE RECOMMANDÉE:

WITH filtered_transactions AS (
  -- Filtrer les transactions selon les critères
  SELECT * FROM raw_transactions
  WHERE user_id = :user_id
    AND [autres filtres basés sur l'intention]
),
search_summary AS (
  -- Résumé de la recherche
  SELECT
    COUNT(*) as total_results,
    COALESCE(SUM(amount), 0) as amount_total,
    COALESCE(AVG(amount), 0) as amount_avg,
    COALESCE(MIN(amount), 0) as amount_min,
    COALESCE(MAX(amount), 0) as amount_max
  FROM filtered_transactions
),
aggregations AS (
  -- Agrégations groupées
  SELECT
    json_build_object(
      'by_period', (SELECT json_agg(row_to_json(t)) FROM (...) t),
      'by_category', (SELECT json_agg(row_to_json(t)) FROM (...) t),
      'by_merchant', (SELECT json_agg(row_to_json(t)) FROM (...) t),
      'by_payment', (SELECT json_agg(row_to_json(t)) FROM (...) t)
    ) as data
  FROM filtered_transactions
),
top_transactions AS (
  -- Top 50 transactions
  SELECT * FROM filtered_transactions
  ORDER BY transaction_date DESC
  LIMIT 50
)
SELECT
  (SELECT row_to_json(s) FROM search_summary s) as search_summary,
  (SELECT data FROM aggregations) as aggregations,
  (SELECT json_agg(row_to_json(t)) FROM top_transactions t) as top_50_transactions;

IMPORTANT:
- La requête finale doit retourner UNE SEULE ligne avec 3 colonnes JSON
- Toujours inclure WHERE user_id = :user_id
- Gérer les cas où filtered_transactions est vide
"""

SQL_GENERATION_USER_PROMPT = """
Génère une requête SQL PostgreSQL basée sur cette intention:

{intent_json}

Date actuelle: {current_date}

La requête doit:
1. Filtrer par user_id = :user_id (OBLIGATOIRE)
2. Appliquer tous les filtres de l'intention (time_periods, categories, merchants, amount_filters)
3. Calculer le résumé de recherche
4. Générer les agrégations demandées
5. Retourner les top 50 transactions

Réponds UNIQUEMENT avec la requête SQL, sans texte avant ni après.
"""


class SQLGenerator:
    """Generator for SQL queries using DeepSeek API."""

    def __init__(self):
        """Initialize the SQL generator."""
        self.client = deepseek_client

    async def generate(self, intent: IntentAnalysis, current_date: str = None) -> str:
        """
        Generate SQL query from intent analysis.

        Args:
            intent: Analyzed intent from user query
            current_date: Current date for time period calculations (optional)

        Returns:
            str: Generated SQL query

        Raises:
            Exception: If API call fails
        """
        from datetime import datetime

        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")

        # Get actual database schema
        db_schema = await get_database_schema()

        # Prepare prompts
        system_prompt = SQL_GENERATION_SYSTEM_PROMPT.format(db_schema=db_schema)
        user_prompt = SQL_GENERATION_USER_PROMPT.format(
            intent_json=intent.model_dump_json(indent=2),
            current_date=current_date
        )

        # Call DeepSeek API
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2048
        )

        # Extract SQL query
        sql_query = response.choices[0].message.content.strip()

        # Clean up markdown code blocks if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

        return sql_query.strip()
