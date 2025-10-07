#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
apt-get install -y unzip
unzip awscliv2.zip
./aws/install

# Install CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/arm64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Create app directory
mkdir -p /opt/harena
cd /opt/harena

# Create .env file for backend
cat > /opt/harena/.env << 'ENV_FILE'
# Database
DATABASE_URL=postgresql://${db_username}:${db_password}@${db_host}/${db_name}
POSTGRES_SERVER=${db_host}
POSTGRES_DB=${db_name}
POSTGRES_USER=${db_username}
POSTGRES_PASSWORD=${db_password}

# Redis
REDIS_URL=rediss://default:${redis_auth_token}@${redis_endpoint}:6379
REDIS_AUTH_TOKEN=${redis_auth_token}

# Elasticsearch (local Docker)
SEARCHBOX_URL=http://localhost:9200

# DeepSeek API
DEEPSEEK_API_KEY=${deepseek_api_key}

# App config
SECRET_KEY=${secret_key}
ENVIRONMENT=${environment}
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=*
ENV_FILE

# Create docker-compose.yml
cat > /opt/harena/docker-compose.yml << 'COMPOSE_FILE'
version: '3.8'

services:
  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: harena-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms256m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    restart: always
    mem_limit: 768m

  backend:
    image: BACKEND_IMAGE_PLACEHOLDER
    container_name: harena-backend
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      - elasticsearch
    restart: always
    command: uvicorn local_app:app --host 0.0.0.0 --port 8000

volumes:
  es_data:
COMPOSE_FILE

# Start Elasticsearch only (backend will be deployed separately)
docker-compose up -d elasticsearch

# Wait for Elasticsearch to be ready
echo "Waiting for Elasticsearch..."
for i in {1..30}; do
  if curl -s http://localhost:9200 > /dev/null; then
    echo "Elasticsearch is ready!"
    break
  fi
  sleep 2
done

# Setup CloudWatch Logs
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CW_CONFIG'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/syslog",
            "log_group_name": "/aws/ec2/harena",
            "log_stream_name": "{instance_id}/syslog"
          }
        ]
      }
    }
  }
}
CW_CONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

echo "EC2 setup complete!"
