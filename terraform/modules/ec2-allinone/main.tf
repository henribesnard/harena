# EC2 All-in-One avec Docker Compose
# Héberge: Backend, PostgreSQL, Redis, Elasticsearch

# Security Group
resource "aws_security_group" "allinone" {
  name        = "harena-allinone-sg-${var.environment}"
  description = "Security group for all-in-one EC2"
  vpc_id      = var.vpc_id

  # HTTP pour le backend API
  ingress {
    from_port   = 8000
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Backend API"
  }

  # PostgreSQL pour accès dev (DBeaver, etc.)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "PostgreSQL database access"
  }

  # SSH via SSM uniquement (pas de port 22 ouvert)

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "harena-allinone-sg-${var.environment}"
  }
}

# IAM Role
resource "aws_iam_role" "ec2" {
  name = "harena-ec2-allinone-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "harena-ec2-allinone-role-${var.environment}"
  }
}

# SSM Access
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# CloudWatch Logs
resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# ECR Access Policy
resource "aws_iam_policy" "ecr_access" {
  name        = "harena-ec2-ecr-access-${var.environment}"
  description = "Allow EC2 to pull images from ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.backup_s3_bucket}",
          "arn:aws:s3:::${var.backup_s3_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ecr_access.arn
}

# Instance Profile
resource "aws_iam_instance_profile" "ec2" {
  name = "harena-ec2-allinone-profile-${var.environment}"
  role = aws_iam_role.ec2.name
}

# Ubuntu 22.04 ARM64 AMI
data "aws_ami" "ubuntu_arm" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# User data pour installer Docker et Docker Compose
locals {
  user_data = templatefile("${path.module}/user_data.sh", {
    db_name          = var.db_name
    db_username      = var.db_username
    db_password      = var.db_password
    redis_auth_token = var.redis_auth_token
    deepseek_api_key = var.deepseek_api_key
    secret_key       = var.secret_key
    environment      = var.environment
  })
}

# EC2 Instance
resource "aws_instance" "allinone" {
  ami           = data.aws_ami.ubuntu_arm.id
  instance_type = var.instance_type

  # Spot ou On-Demand
  instance_market_options {
    market_type = var.use_spot_instances ? "spot" : null

    dynamic "spot_options" {
      for_each = var.use_spot_instances ? [1] : []
      content {
        max_price          = var.spot_max_price
        spot_instance_type = "one-time"
      }
    }
  }

  subnet_id                   = var.public_subnet_id
  vpc_security_group_ids      = [aws_security_group.allinone.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2.name
  associate_public_ip_address = true

  root_block_device {
    volume_size           = var.ebs_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = local.user_data

  tags = {
    Name = "harena-allinone-${var.environment}"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Elastic IP (optionnel mais recommandé)
resource "aws_eip" "allinone" {
  instance = aws_instance.allinone.id
  domain   = "vpc"

  tags = {
    Name = "harena-allinone-eip-${var.environment}"
  }
}
