"""Module 5: Context Builder for LLM consumption."""

from typing import Dict, Any, List


class ContextBuilder:
    """Builder for structured context from SQL results."""

    MAX_CONTEXT_TOKENS = 2000

    def __init__(self):
        """Initialize the context builder."""
        pass

    def build(self, sql_results: Dict[str, Any]) -> str:
        """
        Build structured context from SQL results.

        Args:
            sql_results: Results from SQL execution

        Returns:
            str: Formatted context for LLM (< 2000 tokens)
        """
        context_parts = []

        # BLOC 1: Search Summary (~150 tokens)
        summary = self._build_summary(sql_results.get('search_summary', {}))
        context_parts.append("=== RÉSUMÉ DE RECHERCHE ===")
        context_parts.append(summary)

        # BLOC 2: Aggregations (~600 tokens)
        aggregations = self._build_aggregations(sql_results.get('aggregations', {}))
        if aggregations:
            context_parts.append("\n=== AGRÉGATIONS ===")
            context_parts.append(aggregations)

        # BLOC 3: Top 50 Transactions (~1000 tokens)
        transactions = self._build_transactions(sql_results.get('top_50_transactions', []))
        if transactions:
            context_parts.append("\n=== TOP 50 TRANSACTIONS ===")
            context_parts.append(transactions)

        # BLOC 4: Metadata (~50 tokens)
        metadata = self._build_metadata(sql_results)
        context_parts.append("\n=== MÉTADONNÉES ===")
        context_parts.append(metadata)

        full_context = "\n".join(context_parts)

        # Log context size for debugging
        context_size_chars = len(full_context)
        context_size_tokens = context_size_chars // 4  # Rough estimate: 1 token ≈ 4 chars
        num_transactions = len(sql_results.get('top_50_transactions', []))
        print(f"[Context Builder] Context size: {context_size_chars} chars (~{context_size_tokens} tokens)")
        print(f"[Context Builder] Transactions included: {num_transactions}")

        return full_context

    def _build_summary(self, summary: Dict[str, Any]) -> str:
        """Build search summary block."""
        if not summary:
            return "Aucun résultat trouvé."

        total = summary.get('total_results', 0)
        amount_total = summary.get('amount_total', 0)
        amount_avg = summary.get('amount_avg', 0)
        amount_min = summary.get('amount_min', 0)
        amount_max = summary.get('amount_max', 0)

        lines = [
            f"Nombre de transactions: {total}",
            f"Montant total: {amount_total:.2f}€",
            f"Montant moyen: {amount_avg:.2f}€",
            f"Montant minimum: {amount_min:.2f}€",
            f"Montant maximum: {amount_max:.2f}€"
        ]

        return "\n".join(lines)

    def _build_aggregations(self, aggregations: Dict[str, Any]) -> str:
        """Build aggregations block."""
        if not aggregations:
            return ""

        parts = []

        # By period
        by_period = aggregations.get('by_period', [])
        if by_period:
            parts.append("Par période:")
            for item in by_period[:10]:  # Limit to 10
                period = item.get('period', 'N/A')
                amount = item.get('amount_total', 0)
                count = item.get('transaction_count', 0)
                parts.append(f"  - {period}: {amount:.2f}€ ({count} transactions)")

        # By category
        by_category = aggregations.get('by_category', [])
        if by_category:
            parts.append("\nPar catégorie:")
            for item in by_category[:10]:
                category = item.get('category', 'N/A')
                amount = item.get('amount_total', 0)
                count = item.get('transaction_count', 0)
                parts.append(f"  - {category}: {amount:.2f}€ ({count} transactions)")

        # By merchant
        by_merchant = aggregations.get('by_merchant', [])
        if by_merchant:
            parts.append("\nPar commerce:")
            for item in by_merchant[:10]:
                merchant = item.get('merchant', 'N/A')
                amount = item.get('amount_total', 0)
                count = item.get('transaction_count', 0)
                parts.append(f"  - {merchant}: {amount:.2f}€ ({count} transactions)")

        # By payment method
        by_payment = aggregations.get('by_payment', [])
        if by_payment:
            parts.append("\nPar mode de paiement:")
            for item in by_payment[:10]:
                method = item.get('payment_type', 'N/A')  # Changed from payment_method to payment_type
                amount = item.get('amount_total', 0)
                count = item.get('transaction_count', 0)
                parts.append(f"  - {method}: {amount:.2f}€ ({count} transactions)")

        return "\n".join(parts)

    def _build_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Build top transactions block."""
        if not transactions:
            return ""

        lines = []
        for idx, txn in enumerate(transactions[:50], 1):  # Limit to 50
            date = txn.get('date', txn.get('transaction_date', 'N/A'))
            amount = txn.get('amount', 0)
            merchant = txn.get('merchant_name', txn.get('merchant', 'N/A'))
            category = txn.get('category_name', txn.get('category', 'N/A'))
            payment = txn.get('operation_type', txn.get('payment_method', 'N/A'))

            lines.append(
                f"{idx}. {date} | {amount:.2f}€ | {merchant} | {category} | {payment}"
            )

        return "\n".join(lines)

    def _build_metadata(self, sql_results: Dict[str, Any]) -> str:
        """Build metadata block."""
        metadata = sql_results.get('metadata', {})
        total_results = sql_results.get('search_summary', {}).get('total_results', 0)
        transactions_shown = len(sql_results.get('top_50_transactions', []))

        lines = [
            f"Affichage: {transactions_shown} sur {total_results} transactions",
            f"Cache: {'Oui' if metadata.get('cached', False) else 'Non'}"
        ]

        return "\n".join(lines)
