# Security Group for Backend
resource "aws_security_group" "backend" {
  name        = "harena-backend-sg-${var.environment}"
  description = "Security group for backend EC2"
  vpc_id      = var.vpc_id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  # User Service
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "User Service API"
  }

  # Conversation Service
  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Conversation Service API"
  }

  # Sync Service
  ingress {
    from_port   = 8002
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Sync Service API"
  }

  # Enrichment Service
  ingress {
    from_port   = 8003
    to_port     = 8003
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Enrichment Service API"
  }

  # Metric Service
  ingress {
    from_port   = 8004
    to_port     = 8004
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Metric Service API"
  }

  # Search Service
  ingress {
    from_port   = 8005
    to_port     = 8005
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Search Service API"
  }

  # Elasticsearch (internal only)
  ingress {
    from_port = 9200
    to_port   = 9200
    protocol  = "tcp"
    self      = true
    description = "Elasticsearch internal"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "harena-backend-sg-${var.environment}"
  }
}

# IAM Role for EC2
resource "aws_iam_role" "ec2" {
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

# Attach SSM policy for remote access
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Attach CloudWatch Logs policy
resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# Attach S3 Read-Only policy for deployment
resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

# Instance Profile
resource "aws_iam_instance_profile" "ec2" {
  name = "harena-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2.name
}

# Latest Ubuntu 22.04 ARM64 AMI
data "aws_ami" "ubuntu_arm" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# User data script
locals {
  user_data = templatefile("${path.module}/user_data.sh", {
    db_host          = var.db_host
    db_name          = var.db_name
    db_username      = var.db_username
    db_password      = var.db_password
    redis_endpoint   = var.redis_endpoint
    redis_auth_token = var.redis_auth_token
    deepseek_api_key = var.deepseek_api_key
    secret_key       = var.secret_key
    environment      = var.environment
  })
}

# EC2 Instance (Spot or On-Demand)
resource "aws_instance" "backend" {
  ami           = data.aws_ami.ubuntu_arm.id
  instance_type = var.instance_type
  key_name      = "harena-deploy-key"

  # Spot instance configuration - persistent pour permettre stop/start
  instance_market_options {
    market_type = var.use_spot_instances ? "spot" : null

    dynamic "spot_options" {
      for_each = var.use_spot_instances ? [1] : []
      content {
        max_price                      = var.spot_max_price
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  subnet_id                   = var.public_subnet_ids[0]
  vpc_security_group_ids      = [aws_security_group.backend.id]
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
    Name = "harena-backend-${var.environment}"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Elastic IP for Backend (fixed IP address)
resource "aws_eip" "backend" {
  domain = "vpc"

  tags = {
    Name        = "harena-backend-eip-${var.environment}"
    Project     = "harena"
    Environment = var.environment
  }
}

# Associate Elastic IP with EC2 Instance
resource "aws_eip_association" "backend" {
  instance_id   = aws_instance.backend.id
  allocation_id = aws_eip.backend.id
}
