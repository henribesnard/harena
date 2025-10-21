"""Main orchestrator service for conversation processing."""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import time

from ..core.intent_analyzer import IntentAnalyzer
from ..core.sql_generator import SQLGenerator
from ..core.sql_validator import SQLValidator, ValidationError
from ..core.sql_executor import SQLExecutor
from ..core.context_builder import ContextBuilder
from ..core.response_generator import ResponseGenerator
from ..models.responses.conversation_responses import (
    ConversationResponse,
    ResponseMetadata,
    TokenUsage,
    Visualization
)


class ConversationService:
    """
    Main service orchestrating the conversation processing pipeline.

    Pipeline:
    1. Analyze user intent (IntentAnalyzer)
    2. Generate SQL query (SQLGenerator)
    3. Validate SQL (SQLValidator)
    4. Execute SQL with caching (SQLExecutor)
    5. Build context (ContextBuilder)
    6. Generate natural language response (ResponseGenerator)
    """

    def __init__(self):
        """Initialize the conversation service."""
        self.intent_analyzer = IntentAnalyzer()
        self.sql_generator = SQLGenerator()
        self.sql_validator = SQLValidator()
        self.sql_executor = SQLExecutor()
        self.context_builder = ContextBuilder()
        self.response_generator = ResponseGenerator()

    async def process_conversation(
        self,
        user_id: int,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationResponse:
        """
        Process a conversation request end-to-end.

        Args:
            user_id: User ID making the request
            query: User question in natural language
            context: Optional conversation context

        Returns:
            ConversationResponse: Complete response with answer and metadata

        Raises:
            Exception: If any step in the pipeline fails
        """
        start_time = time.time()

        # Generate or retrieve conversation_id
        if context and isinstance(context, dict) and context.get('conversation_id'):
            conversation_id = context.get('conversation_id')
        else:
            conversation_id = str(uuid.uuid4())

        print(f"[Service] Generated conversation_id at start: {conversation_id}")
        print(f"[Service] Type: {type(conversation_id)}")

        try:
            # STEP 1: Analyze intent
            step1_start = time.time()
            intent = await self.intent_analyzer.analyze(query)
            step1_duration = (time.time() - step1_start) * 1000
            print(f"[TIMING] Step 1 (Intent Analysis): {step1_duration:.2f}ms")

            # STEP 2: Generate SQL
            step2_start = time.time()
            sql_query = await self.sql_generator.generate(intent)
            step2_duration = (time.time() - step2_start) * 1000
            print(f"[TIMING] Step 2 (SQL Generation): {step2_duration:.2f}ms")

            # STEP 3: Validate SQL
            step3_start = time.time()
            validation_result = self.sql_validator.validate(sql_query, user_id)
            step3_duration = (time.time() - step3_start) * 1000
            print(f"[TIMING] Step 3 (SQL Validation): {step3_duration:.2f}ms")

            # STEP 4: Execute SQL
            step4_start = time.time()
            sql_results = await self.sql_executor.execute(
                sql_query=sql_query,
                user_id=user_id
            )
            step4_duration = (time.time() - step4_start) * 1000
            print(f"[TIMING] Step 4 (SQL Execution): {step4_duration:.2f}ms")

            # STEP 5: Build context for LLM
            step5_start = time.time()
            llm_context = self.context_builder.build(sql_results)
            step5_duration = (time.time() - step5_start) * 1000
            print(f"[TIMING] Step 5 (Context Building): {step5_duration:.2f}ms")

            # STEP 6: Generate natural language response
            step6_start = time.time()
            response_data = await self.response_generator.generate(
                user_query=query,
                context=llm_context
            )
            step6_duration = (time.time() - step6_start) * 1000
            print(f"[TIMING] Step 6 (Response Generation): {step6_duration:.2f}ms")

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            print(f"[TIMING] TOTAL: {execution_time_ms}ms")
            print(f"[TIMING] Breakdown: Intent={step1_duration:.0f}ms, SQL={step2_duration:.0f}ms, Validate={step3_duration:.0f}ms, Execute={step4_duration:.0f}ms, Context={step5_duration:.0f}ms, Response={step6_duration:.0f}ms")

            # Calculate cost (DeepSeek pricing)
            tokens_used = response_data.get('tokens_used', {})
            input_tokens = tokens_used.get('input', 0)
            output_tokens = tokens_used.get('output', 0)

            # DeepSeek pricing: $0.27/1M input, $1.10/1M output
            cost_usd = (input_tokens / 1_000_000 * 0.27) + (output_tokens / 1_000_000 * 1.10)

            # Build metadata
            metadata = ResponseMetadata(
                execution_time_ms=execution_time_ms,
                tokens_used=TokenUsage(
                    input=input_tokens,
                    output=output_tokens,
                    total=input_tokens + output_tokens
                ),
                cost_usd=round(cost_usd, 6),
                sql_query=sql_query,
                total_transactions_found=sql_results.get('search_summary', {}).get('total_results', 0),
                cached=sql_results.get('metadata', {}).get('cached', False),
                model_used="deepseek-chat"
            )

            # Build visualization (optional, based on intent)
            visualization = self._generate_visualization(
                intent=intent,
                sql_results=sql_results
            )

            # Debug logging before creating response
            print(f"[Service] About to create ConversationResponse")
            print(f"[Service] conversation_id: {conversation_id}")
            print(f"[Service] user_id: {user_id}")
            print(f"[Service] answer: {response_data.get('answer', '')[:100] if response_data.get('answer') else 'NO ANSWER'}")
            print(f"[Service] insights: {response_data.get('insights', [])}")
            print(f"[Service] recommendations: {response_data.get('recommendations', [])}")
            print(f"[Service] visualization: {visualization}")
            print(f"[Service] metadata type: {type(metadata)}")

            # Build final response
            try:
                resp = ConversationResponse(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    answer=response_data.get('answer', ''),
                    insights=response_data.get('insights', []),
                    recommendations=response_data.get('recommendations', []),
                    visualization=visualization,
                    metadata=metadata
                )
                print(f"[Service] ConversationResponse created successfully!")
                return resp
            except Exception as e:
                print(f"[Service] ERROR creating ConversationResponse: {e}")
                print(f"[Service] Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise

        except ValidationError as e:
            # Handle validation errors
            raise Exception(f"SQL validation failed: {str(e)}")

        except Exception as e:
            # Handle other errors
            raise Exception(f"Error processing conversation: {str(e)}")

    def _generate_visualization(
        self,
        intent: Any,
        sql_results: Dict[str, Any]
    ) -> Optional[Visualization]:
        """
        Generate visualization data based on intent and results.

        Args:
            intent: Analyzed intent
            sql_results: SQL execution results

        Returns:
            Visualization or None
        """
        # For now, return None - visualization logic can be added later
        # This would analyze the intent type and aggregations to create appropriate charts
        return None

    async def cleanup(self):
        """Cleanup resources (close connections, etc.)."""
        await self.sql_executor.close_pool()
