output "instance_id" {
  description = "EC2 instance ID"
  value       = module.ec2_allinone.instance_id
}

output "public_ip" {
  description = "Elastic IP address"
  value       = module.ec2_allinone.public_ip
}

output "database_endpoint" {
  description = "PostgreSQL connection endpoint"
  value       = "${module.ec2_allinone.public_ip}:5432"
}

output "redis_endpoint" {
  description = "Redis connection endpoint"
  value       = "${module.ec2_allinone.public_ip}:6379"
}

output "elasticsearch_endpoint" {
  description = "Elasticsearch connection endpoint"
  value       = "http://${module.ec2_allinone.public_ip}:9200"
}

output "backend_url" {
  description = "Backend API URL"
  value       = "http://${module.ec2_allinone.public_ip}:8000"
}

output "frontend_url" {
  description = "Frontend URL"
  value       = "http://${module.ec2_allinone.public_ip}"
}

output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = "aws ssm start-session --target ${module.ec2_allinone.instance_id} --region ${var.aws_region}"
}

output "dbeaver_connection" {
  description = "DBeaver connection string"
  value       = "postgresql://${var.db_username}@${module.ec2_allinone.public_ip}:5432/${var.db_name}"
  sensitive   = true
}
