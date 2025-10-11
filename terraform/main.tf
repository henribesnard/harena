terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend S3 pour stocker le state Terraform (à activer après premier apply)
  # backend "s3" {
  #   bucket         = "harena-terraform-state"
  #   key            = "dev/terraform.tfstate"
  #   region         = "eu-west-1"
  #   encrypt        = true
  #   dynamodb_table = "harena-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Harena"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "Demo"
    }
  }
}

# VPC et réseau
module "vpc" {
  source = "./modules/vpc"

  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds"

  environment          = var.environment
  instance_class       = var.rds_instance_class
  allocated_storage    = var.rds_allocated_storage
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  public_subnet_ids   = module.vpc.public_subnet_ids
  backend_sg_id       = module.ec2.backend_security_group_id
}

# ElastiCache Redis
module "redis" {
  source = "./modules/redis"

  environment        = var.environment
  node_type         = var.redis_node_type
  auth_token        = var.redis_auth_token
  vpc_id            = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  backend_sg_id     = module.ec2.backend_security_group_id
}

# EC2 Backend + Elasticsearch
module "ec2" {
  source = "./modules/ec2"

  environment           = var.environment
  instance_type        = var.ec2_instance_type
  use_spot_instances   = var.use_spot_instances
  spot_max_price       = var.spot_max_price
  ebs_volume_size      = var.ebs_volume_size
  vpc_id               = module.vpc.vpc_id
  public_subnet_ids    = module.vpc.public_subnet_ids

  # Variables d'environnement pour l'application
  db_host              = module.rds.db_endpoint
  db_name              = var.db_name
  db_username          = var.db_username
  db_password          = var.db_password
  redis_endpoint       = module.redis.redis_endpoint
  redis_auth_token     = var.redis_auth_token
  deepseek_api_key     = var.deepseek_api_key
  secret_key           = var.secret_key
}

# S3 + CloudFront pour le frontend + API proxy
module "s3_cloudfront" {
  source = "./modules/s3-cloudfront"

  environment       = var.environment
  domain_name       = var.domain_name
  backend_public_ip = module.ec2.public_ip
}

# OpenSearch (Elasticsearch) - Désactivé: nécessite subscription AWS
# module "opensearch" {
#   source = "./modules/opensearch"
#
#   environment                = var.environment
#   region                     = var.aws_region
#   vpc_id                     = module.vpc.vpc_id
#   private_subnet_ids         = module.vpc.private_subnet_ids
#   backend_security_group_id  = module.ec2.backend_security_group_id
#   allowed_cidr_blocks        = [module.vpc.vpc_cidr]
# }

# Monitoring et Auto-shutdown
module "monitoring" {
  source = "./modules/monitoring"

  environment           = var.environment
  enable_auto_shutdown  = var.enable_auto_shutdown
  shutdown_night       = var.shutdown_night
  shutdown_weekend     = var.shutdown_weekend
  ec2_instance_id      = module.ec2.instance_id
  alert_email          = var.alert_email
}
