output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_endpoint
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.redis.redis_endpoint
}

output "ec2_instance_id" {
  description = "EC2 instance ID"
  value       = module.ec2.instance_id
}

output "ec2_public_ip" {
  description = "EC2 Elastic IP (fixed IP address)"
  value       = module.ec2.public_ip
}

output "elastic_ip_allocation_id" {
  description = "Elastic IP allocation ID"
  value       = module.ec2.elastic_ip_allocation_id
}

output "backend_url" {
  description = "Backend API URL"
  value       = "http://${module.ec2.public_ip}:8000"
}

output "s3_bucket_name" {
  description = "S3 bucket name for frontend"
  value       = module.s3_cloudfront.s3_bucket_name
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain"
  value       = module.s3_cloudfront.cloudfront_domain
}

output "frontend_url" {
  description = "Frontend URL"
  value       = "https://${module.s3_cloudfront.cloudfront_domain}"
}

output "api_cloudfront_domain" {
  description = "API CloudFront domain (HTTPS proxy to backend)"
  value       = module.s3_cloudfront.api_cloudfront_domain
}

output "api_url" {
  description = "API URL (HTTPS via CloudFront)"
  value       = "https://${module.s3_cloudfront.api_cloudfront_domain}"
}

output "ssh_command" {
  description = "SSH command to connect to EC2 (via SSM)"
  value       = "aws ssm start-session --target ${module.ec2.instance_id} --region ${var.aws_region}"
}

# OpenSearch outputs désactivés - module désactivé
# output "opensearch_endpoint" {
#   description = "OpenSearch endpoint URL"
#   value       = module.opensearch.endpoint
# }
#
# output "opensearch_domain_name" {
#   description = "OpenSearch domain name"
#   value       = module.opensearch.domain_name
# }
#
# output "opensearch_kibana_url" {
#   description = "OpenSearch Dashboards (Kibana) URL"
#   value       = module.opensearch.kibana_endpoint
# }
