variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"
}

variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
}

# Network
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# RDS
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t4g.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "harena"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "harena_admin"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Redis
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t4g.micro"
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
}

# EC2
variable "ec2_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t4g.micro"
}

variable "use_spot_instances" {
  description = "Use Spot instances for EC2"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for Spot instances"
  type        = string
  default     = "0.0042"
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 20
}

# Application
variable "deepseek_api_key" {
  description = "DeepSeek API key"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "Application secret key"
  type        = string
  sensitive   = true
}

# Frontend
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# Monitoring
variable "enable_auto_shutdown" {
  description = "Enable auto-shutdown for EC2"
  type        = bool
  default     = true
}

variable "shutdown_night" {
  description = "Shutdown EC2 at night"
  type        = bool
  default     = true
}

variable "shutdown_weekend" {
  description = "Shutdown EC2 on weekends"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email for CloudWatch alerts"
  type        = string
}
