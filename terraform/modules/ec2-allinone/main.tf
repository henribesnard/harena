# Security Group for All-in-One Instance
resource "aws_security_group" "allinone" {
  name        = "harena-allinone-sg-${var.environment}"
  description = "Security group for Harena all-in-one instance"
  vpc_id      = var.vpc_id

  # SSH via SSM (no inbound needed)

  # PostgreSQL - accessible from your IP only
  ingress {
    description = "PostgreSQL from allowed IP"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  # Redis - accessible from your IP only
  ingress {
    description = "Redis from allowed IP"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  # Elasticsearch - accessible from your IP only
  ingress {
    description = "Elasticsearch from allowed IP"
    from_port   = 9200
    to_port     = 9200
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  # Uptime Kuma - accessible from your IP only
  ingress {
    description = "Uptime Kuma Monitoring from allowed IP"
    from_port   = 3100
    to_port     = 3100
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  # Backend API - public access
  ingress {
    description = "Backend API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Frontend HTTP - public access
  ingress {
    description = "Frontend HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Frontend HTTPS - public access
  ingress {
    description = "Frontend HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound - allow all
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "harena-allinone-sg-${var.environment}"
  }
}

# IAM Role for EC2 (SSM access)
resource "aws_iam_role" "ec2_role" {
  name = "harena-ec2-role-${var.environment}"

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
    Name = "harena-ec2-role-${var.environment}"
  }
}

# Attach SSM policy
resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance Profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "harena-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_role.name

  tags = {
    Name = "harena-ec2-profile-${var.environment}"
  }
}

# Get latest Amazon Linux 2023 ARM64 AMI
data "aws_ami" "amazon_linux_2023_arm" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

# User Data Script
locals {
  user_data = templatefile("${path.module}/user-data.sh", {
    db_name          = var.db_name
    db_username      = var.db_username
    db_password      = var.db_password
    redis_auth_token = var.redis_auth_token
    secret_key       = var.secret_key
    deepseek_api_key = var.deepseek_api_key
  })
}

# EC2 Instance
resource "aws_instance" "allinone" {
  ami                    = data.aws_ami.amazon_linux_2023_arm.id
  instance_type          = var.instance_type
  subnet_id              = var.public_subnet_id
  vpc_security_group_ids = [aws_security_group.allinone.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.ebs_volume_size
    delete_on_termination = false
    encrypted             = true

    tags = {
      Name = "harena-allinone-root-${var.environment}"
    }
  }

  user_data = local.user_data

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tags = {
    Name         = "harena-allinone-${var.environment}"
    Architecture = "All-in-One"
  }
}

# Elastic IP
resource "aws_eip" "allinone" {
  domain = "vpc"

  tags = {
    Name = "harena-allinone-eip-${var.environment}"
  }
}

# Associate Elastic IP
resource "aws_eip_association" "allinone" {
  instance_id   = aws_instance.allinone.id
  allocation_id = aws_eip.allinone.id
}
