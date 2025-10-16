#!/bin/bash
set -e

# Logs
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Starting Harena All-in-One Setup ==="
date

# Update system
echo "Updating system packages..."
dnf update -y

# Install Docker
echo "Installing Docker..."
dnf install -y docker
systemctl enable docker
systemctl start docker

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="2.24.5"
curl -L "https://github.com/docker/compose/releases/download/v$${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
echo "Creating application directory..."
mkdir -p /opt/harena
cd /opt/harena

# Create Docker Compose configuration
echo "Creating Docker Compose configuration..."
cat > docker-compose.yml <<'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: harena-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${db_name}
      POSTGRES_USER: ${db_username}
      POSTGRES_PASSWORD: ${db_password}
      POSTGRES_INITDB_ARGS: "-E UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres-init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    command: >
      postgres
      -c listen_addresses='*'
      -c max_connections=100
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=2621kB
      -c min_wal_size=1GB
      -c max_wal_size=4GB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${db_username} -d ${db_name}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: harena-redis
    restart: unless-stopped
    command: >
      redis-server
      --requirepass ${redis_auth_token}
      --appendonly yes
      --appendfsync everysec
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: harena-elasticsearch
    restart: unless-stopped
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - bootstrap.memory_lock=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  elasticsearch_data:
    driver: local
EOF

# Create PostgreSQL init script
echo "Creating PostgreSQL initialization script..."
cat > postgres-init.sql <<'EOSQL'
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ${db_name} TO ${db_username};
EOSQL

# Create environment file for backend (when deployed)
echo "Creating environment template..."
cat > .env.template <<'ENVEOF'
# Database
DATABASE_URL=postgresql://${db_username}:${db_password}@localhost:5432/${db_name}

# Redis
REDIS_URL=redis://:${redis_auth_token}@localhost:6379/0

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# Application
SECRET_KEY=${secret_key}
DEEPSEEK_API_KEY=${deepseek_api_key}
ENVIRONMENT=production
ENVEOF

# Start Docker Compose services
echo "Starting Docker Compose services..."
/usr/local/bin/docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose ps

# Show service logs
echo "=== PostgreSQL logs ==="
docker-compose logs postgres | tail -20

echo "=== Redis logs ==="
docker-compose logs redis | tail -20

echo "=== Elasticsearch logs ==="
docker-compose logs elasticsearch | tail -20

# Create systemd service for auto-start
echo "Creating systemd service..."
cat > /etc/systemd/system/harena-stack.service <<'SYSTEMDEOF'
[Unit]
Description=Harena All-in-One Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/harena
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
SYSTEMDEOF

systemctl daemon-reload
systemctl enable harena-stack.service

echo "=== Setup Complete ==="
echo "PostgreSQL: localhost:5432"
echo "Redis: localhost:6379"
echo "Elasticsearch: localhost:9200"
echo "All services are running!"
date
