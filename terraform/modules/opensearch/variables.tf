variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "vpc_id" {
  description = "VPC ID where OpenSearch will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for OpenSearch"
  type        = list(string)
}

variable "backend_security_group_id" {
  description = "Security group ID of backend EC2 instances"
  type        = string
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access OpenSearch"
  type        = list(string)
  default     = ["10.0.0.0/16"] # VPC CIDR by default
}
