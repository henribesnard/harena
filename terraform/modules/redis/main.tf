# Security Group for Redis
resource "aws_security_group" "redis" {
  name        = "harena-redis-sg-${var.environment}"
  description = "Security group for ElastiCache Redis"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.backend_sg_id]
    description     = "Redis from backend"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "harena-redis-sg-${var.environment}"
  }
}

# Redis Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "harena-redis-subnet-${var.environment}"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "harena-redis-subnet-${var.environment}"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "harena-redis-${var.environment}"
  engine               = "redis"
  engine_version       = "7.0"
  node_type           = var.node_type
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379

  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  # Authentication
  auth_token                 = var.auth_token
  transit_encryption_enabled = true

  # Free Tier optimizations
  snapshot_retention_limit = 0

  tags = {
    Name = "harena-redis-${var.environment}"
  }
}
