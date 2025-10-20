"""Module 4: SQL Execution with Redis caching."""

import asyncpg
import redis
import json
import hashlib
from typing import Dict, Any, Optional
import os


class SQLExecutor:
    """Executor for SQL queries with caching."""

    def __init__(self):
        """Initialize the SQL executor."""
        # PostgreSQL connection will be created per request
        self.pg_pool: Optional[asyncpg.Pool] = None

        # Redis client for caching
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                # Use REDIS_URL which includes password
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True
                )
            else:
                # Fallback to individual parameters
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD"),
                    db=0,
                    decode_responses=True
                )
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            self.redis_client = None

        self.cache_ttl = 900  # 15 minutes

    async def initialize_pool(self):
        """Initialize PostgreSQL connection pool."""
        if self.pg_pool is None:
            self.pg_pool = await asyncpg.create_pool(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", 5432)),
                database=os.getenv("POSTGRES_DB", "harena"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "postgres"),
                min_size=2,
                max_size=10
            )

    async def close_pool(self):
        """Close PostgreSQL connection pool."""
        if self.pg_pool:
            await self.pg_pool.close()
            self.pg_pool = None

    def _generate_cache_key(self, user_id: int, sql_query: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key from query and parameters.

        Args:
            user_id: User ID
            sql_query: SQL query
            params: Query parameters

        Returns:
            str: Cache key
        """
        # Create a unique hash from user_id + query + params
        content = f"{user_id}:{sql_query}:{json.dumps(params, sort_keys=True)}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"sql_cache:{user_id}:{hash_value}"

    def _replace_named_params(self, sql_query: str, params: Dict[str, Any]) -> tuple[str, list]:
        """
        Replace named parameters (:param) with positional parameters ($1, $2, ...).

        Args:
            sql_query: SQL query with named parameters
            params: Dictionary of parameters

        Returns:
            tuple: (modified SQL query, list of parameter values in order)
        """
        import re

        # Find all named parameters in the query, but NOT PostgreSQL type casts (::type)
        # Use negative lookbehind to avoid matching :: (type cast)
        param_pattern = re.compile(r'(?<!:):(\w+)(?!:)')
        param_names = param_pattern.findall(sql_query)

        # Create ordered list of parameter values
        param_values = []
        param_mapping = {}

        for param_name in param_names:
            if param_name not in param_mapping:
                param_mapping[param_name] = f'${len(param_values) + 1}'
                param_values.append(params.get(param_name))

        # Replace named parameters with positional ones
        # Again use the same pattern to avoid replacing ::type casts
        modified_query = sql_query
        for param_name, positional in param_mapping.items():
            # Use word boundary to ensure we replace the whole parameter name
            modified_query = re.sub(rf'(?<!:):({param_name})(?!:)\b', positional, modified_query)

        return modified_query, param_values

    async def execute(
        self,
        sql_query: str,
        user_id: int,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query with caching.

        Args:
            sql_query: SQL query to execute
            user_id: User ID for RLS
            params: Query parameters

        Returns:
            dict: Query results with metadata

        Raises:
            Exception: If query execution fails
        """
        if params is None:
            params = {}

        # Always include user_id in parameters
        params['user_id'] = user_id

        # Generate cache key (use original query for consistency)
        cache_key = self._generate_cache_key(user_id, sql_query, params)

        # Check cache
        cached_result = None
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cached_result = json.loads(cached_data)
                    cached_result['metadata']['cached'] = True
                    return cached_result
            except Exception as e:
                print(f"Redis cache read error: {e}")

        # Initialize pool if needed
        await self.initialize_pool()

        # Convert named parameters to positional parameters
        modified_query, param_values = self._replace_named_params(sql_query, params)

        # Debug logging
        print(f"[SQL Executor] Original query: {sql_query[:200]}...")
        print(f"[SQL Executor] Modified query: {modified_query[:200]}...")
        print(f"[SQL Executor] Params: {params}")
        print(f"[SQL Executor] Param values: {param_values}")

        # Execute query
        try:
            async with self.pg_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute(f"SET app.current_user_id = {user_id}")

                # Execute query with positional parameters
                row = await conn.fetchrow(modified_query, *param_values)

                # Debug logging
                print(f"[SQL Executor] Row type: {type(row)}")
                print(f"[SQL Executor] Row content: {row}")
                if row:
                    print(f"[SQL Executor] Row keys: {row.keys() if hasattr(row, 'keys') else 'N/A'}")

                if row is None:
                    result = {
                        "search_summary": {},
                        "aggregations": {},
                        "top_50_transactions": [],
                        "metadata": {
                            "cached": False,
                            "rows_returned": 0
                        }
                    }
                else:
                    # PostgreSQL returns JSON as strings, need to parse them
                    # Parse JSON strings to Python objects
                    try:
                        search_summary = row['search_summary']
                        aggregations = row['aggregations']
                        top_50 = row['top_50_transactions']
                    except Exception as e:
                        print(f"[SQL Executor] Error accessing row fields: {e}")
                        print(f"[SQL Executor] Row type: {type(row)}")
                        print(f"[SQL Executor] Row: {row}")
                        raise

                    # If they're strings, parse them
                    if isinstance(search_summary, str):
                        search_summary = json.loads(search_summary) if search_summary else {}
                    if isinstance(aggregations, str):
                        aggregations = json.loads(aggregations) if aggregations else {}
                    if isinstance(top_50, str):
                        top_50 = json.loads(top_50) if top_50 else []

                    result = {
                        "search_summary": search_summary or {},
                        "aggregations": aggregations or {},
                        "top_50_transactions": top_50 or [],
                        "metadata": {
                            "cached": False,
                            "rows_returned": len(top_50) if top_50 else 0
                        }
                    }

                # Debug: log result structure
                print(f"[SQL Executor] Result keys: {result.keys()}")
                print(f"[SQL Executor] search_summary type: {type(result['search_summary'])}")
                print(f"[SQL Executor] search_summary content: {result['search_summary']}")

                # Cache the result
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            cache_key,
                            self.cache_ttl,
                            json.dumps(result)
                        )
                    except Exception as e:
                        print(f"Redis cache write error: {e}")

                return result

        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")
