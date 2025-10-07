variable "environment" {
  description = "Environment name"
  type        = string
}

variable "node_type" {
  description = "ElastiCache node type"
  type        = string
}

variable "auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs"
  type        = list(string)
}

variable "backend_sg_id" {
  description = "Backend security group ID"
  type        = string
}
