output "db_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
}

output "db_arn" {
  description = "RDS ARN"
  value       = aws_db_instance.main.arn
}

output "security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}
