output "redis_endpoint" {
  description = "Redis endpoint"
  value       = aws_elasticache_cluster.main.cache_nodes[0].address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_cluster.main.port
}

output "security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.redis.id
}
