output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.frontend.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.frontend.arn
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain"
  value       = aws_cloudfront_distribution.frontend.domain_name
}

output "cloudfront_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.frontend.id
}

output "api_cloudfront_domain" {
  description = "API CloudFront distribution domain (HTTPS)"
  value       = aws_cloudfront_distribution.api.domain_name
}

output "api_cloudfront_id" {
  description = "API CloudFront distribution ID"
  value       = aws_cloudfront_distribution.api.id
}
