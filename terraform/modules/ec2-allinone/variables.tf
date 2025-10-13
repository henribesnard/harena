variable "environment" {
  description = "Environment name"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t4g.small"
}

variable "use_spot_instances" {
  description = "Use Spot instances"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for Spot instances"
  type        = string
  default     = "0.0084"  # t4g.small spot price
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 40
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "public_subnet_id" {
  description = "Public subnet ID"
  type        = string
}

# Application variables
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

variable "redis_auth_token" {
  description = "Redis authentication token"
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

variable "backup_s3_bucket" {
  description = "S3 bucket for backups"
  type        = string
}
