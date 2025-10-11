# Security Group for RDS
resource "aws_security_group" "rds" {
  name        = "harena-rds-sg-${var.environment}"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.backend_sg_id]
    description     = "PostgreSQL from backend"
  }

  # En dev: accès public pour DBeaver
  dynamic "ingress" {
    for_each = var.environment == "dev" ? [1] : []
    content {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "Public access for development"
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "harena-rds-sg-${var.environment}"
  }
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name = "harena-db-subnet-${var.environment}"
  # En dev: utiliser subnets publics pour accès direct, en prod: subnets privés
  subnet_ids = var.environment == "dev" ? var.public_subnet_ids : var.private_subnet_ids

  tags = {
    Name = "harena-db-subnet-${var.environment}"
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "main" {
  identifier     = "harena-db-${var.environment}"
  engine         = "postgres"
  engine_version = "16"

  instance_class    = var.instance_class
  allocated_storage = var.allocated_storage
  storage_type      = "gp2"
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Free Tier optimizations
  multi_az               = false
  publicly_accessible    = var.environment == "dev" ? true : false
  backup_retention_period = 1
  skip_final_snapshot    = true
  deletion_protection    = false

  # Performance
  performance_insights_enabled = false
  enabled_cloudwatch_logs_exports = []

  tags = {
    Name = "harena-db-${var.environment}"
  }
}
