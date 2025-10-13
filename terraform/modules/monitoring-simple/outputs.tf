output "log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ec2.name
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = length(aws_sns_topic.alerts) > 0 ? aws_sns_topic.alerts[0].arn : ""
}
