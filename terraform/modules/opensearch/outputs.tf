output "domain_id" {
  description = "Domain ID of the OpenSearch cluster"
  value       = aws_opensearch_domain.main.domain_id
}

output "domain_name" {
  description = "Domain name of the OpenSearch cluster"
  value       = aws_opensearch_domain.main.domain_name
}

output "endpoint" {
  description = "Endpoint URL of the OpenSearch cluster"
  value       = "https://${aws_opensearch_domain.main.endpoint}"
}

output "kibana_endpoint" {
  description = "Kibana endpoint URL"
  value       = "https://${aws_opensearch_domain.main.endpoint}/_dashboards/"
}

output "arn" {
  description = "ARN of the OpenSearch domain"
  value       = aws_opensearch_domain.main.arn
}

output "security_group_id" {
  description = "Security group ID for OpenSearch"
  value       = aws_security_group.opensearch.id
}
