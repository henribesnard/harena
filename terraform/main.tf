terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Harena"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "Optimized"
    }
  }
}

# VPC Simple
module "vpc" {
  source = "./modules/vpc"

  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

# EC2 All-in-One (PostgreSQL + Redis + Elasticsearch + Backend + Frontend)
module "ec2_allinone" {
  source = "./modules/ec2-allinone"

  environment        = var.environment
  instance_type      = var.instance_type
  ebs_volume_size    = var.ebs_volume_size
  vpc_id             = module.vpc.vpc_id
  public_subnet_id   = module.vpc.public_subnet_id

  # Database credentials
  db_name            = var.db_name
  db_username        = var.db_username
  db_password        = var.db_password

  # Redis
  redis_auth_token   = var.redis_auth_token

  # Application secrets
  secret_key         = var.secret_key
  deepseek_api_key   = var.deepseek_api_key

  # Access control
  allowed_ip         = var.allowed_ip
}
