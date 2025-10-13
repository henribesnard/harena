# Module OpenSearch (Elasticsearch) pour AWS
# Configuration économique pour environnement dev

resource "aws_opensearch_domain" "main" {
  domain_name    = "harena-search-${var.environment}"
  engine_version = "OpenSearch_2.11"

  cluster_config {
    # Configuration la plus économique : t3.small.search instance
    instance_type  = "t3.small.search"
    instance_count = 1 # Single node pour dev

    # Pas de déploiement multi-AZ pour économiser
    zone_awareness_enabled = false

    # Pas de dedicated master pour économiser
    dedicated_master_enabled = false
  }

  # Stockage EBS économique
  ebs_options {
    ebs_enabled = true
    volume_type = "gp3" # gp3 est plus économique que gp2
    volume_size = 10    # 10GB minimum
    iops        = 3000  # IOPS de base pour gp3
    throughput  = 125   # Throughput de base pour gp3
  }

  # Accès depuis le VPC uniquement (pas d'accès public)
  vpc_options {
    subnet_ids         = [var.private_subnet_ids[0]] # 1 seul subnet car 1 seul node
    security_group_ids = [aws_security_group.opensearch.id]
  }

  # Configuration du domaine
  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"

    # Pas de custom endpoint pour économiser
    custom_endpoint_enabled = false
  }

  # Politique d'accès
  access_policies = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action   = "es:*"
        Resource = "arn:aws:es:${var.region}:${data.aws_caller_identity.current.account_id}:domain/harena-search-${var.environment}/*"
        Condition = {
          IpAddress = {
            "aws:SourceIp" = var.allowed_cidr_blocks
          }
        }
      }
    ]
  })

  # Pas de chiffrement avancé pour économiser
  encrypt_at_rest {
    enabled = false
  }

  # Pas de chiffrement node-to-node pour économiser en dev
  node_to_node_encryption {
    enabled = false
  }

  # Pas de logs CloudWatch pour économiser
  # (on peut activer plus tard si besoin)

  # Advanced options
  advanced_options = {
    "rest.action.multi.allow_explicit_index" = "true"
    "override_main_response_version"         = "false"
  }

  tags = {
    Name        = "harena-search-${var.environment}"
    Environment = var.environment
    ManagedBy   = "terraform"
    Service     = "search"
  }
}

# Security Group pour OpenSearch
resource "aws_security_group" "opensearch" {
  name_prefix = "harena-opensearch-${var.environment}-"
  description = "Security group for OpenSearch domain"
  vpc_id      = var.vpc_id

  # Accès depuis le security group backend (EC2)
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [var.backend_security_group_id]
    description     = "HTTPS from backend"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name        = "harena-opensearch-${var.environment}"
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Data source pour account ID
data "aws_caller_identity" "current" {}
