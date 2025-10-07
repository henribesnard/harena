#!/bin/bash
set -e

echo "🚀 Déploiement Frontend Harena"
echo "==============================="

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Fichier .env introuvable"
    exit 1
fi

# Get S3 bucket and CloudFront from Terraform
cd terraform
S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
CLOUDFRONT_ID=$(terraform output -raw cloudfront_id 2>/dev/null || echo "")
CLOUDFRONT_URL=$(terraform output -raw cloudfront_domain 2>/dev/null || echo "")
BACKEND_URL=$(terraform output -raw backend_url 2>/dev/null || echo "")
cd ..

if [ -z "$S3_BUCKET" ]; then
    echo "❌ Bucket S3 introuvable. Lancez d'abord 'terraform apply'"
    exit 1
fi

echo "📦 Bucket S3: $S3_BUCKET"
echo "🌐 CloudFront: $CLOUDFRONT_URL"

# Build frontend
echo "🔨 Build du frontend..."
cd harena_front

# Create .env.production with backend URL
cat > .env.production << EOF
VITE_API_URL=$BACKEND_URL
VITE_APP_NAME=Harena
EOF

# Install dependencies and build
npm ci --prefer-offline --no-audit
npm run build

# Upload to S3
echo "📤 Upload vers S3..."
aws s3 sync dist/ "s3://$S3_BUCKET/" \
    --delete \
    --cache-control "public, max-age=31536000, immutable" \
    --exclude "index.html"

# Upload index.html separately (no cache)
aws s3 cp dist/index.html "s3://$S3_BUCKET/index.html" \
    --cache-control "no-cache, no-store, must-revalidate" \
    --metadata-directive REPLACE

# Invalidate CloudFront cache
echo "🔄 Invalidation du cache CloudFront..."
aws cloudfront create-invalidation \
    --distribution-id "$CLOUDFRONT_ID" \
    --paths "/*" \
    --region "$AWS_REGION"

cd ..

echo "✅ Frontend déployé avec succès !"
echo "🔗 URL: https://$CLOUDFRONT_URL"
echo ""
echo "⏳ Le cache CloudFront peut prendre 2-3 minutes à se propager"
