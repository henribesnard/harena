variable "environment" {
  description = "Environment name"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "use_spot_instances" {
  description = "Use Spot instances"
  type        = bool
}

variable "spot_max_price" {
  description = "Maximum price for Spot instances"
  type        = string
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs"
  type        = list(string)
}

variable "db_host" {
  description = "Database host"
  type        = string
}

variable "db_name" {
  description = "Database name"
  type        = string
}

variable "db_username" {
  description = "Database username"
  type        = string
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_endpoint" {
  description = "Redis endpoint"
  type        = string
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
}

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
