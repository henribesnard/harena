# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ec2" {
  name              = "/aws/ec2/harena"
  retention_in_days = 7

  tags = {
    Name = "harena-logs-${var.environment}"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "harena-alerts-${var.environment}"

  tags = {
    Name = "harena-alerts-${var.environment}"
  }
}

# SNS Subscription
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# CloudWatch Alarm - Low CPU (auto-stop)
resource "aws_cloudwatch_metric_alarm" "low_cpu" {
  alarm_name          = "harena-low-cpu-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "6"  # 30 minutes (6 x 5 min)
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "5"
  alarm_description   = "Stop EC2 when CPU < 5% for 30 minutes"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    InstanceId = var.ec2_instance_id
  }

  tags = {
    Name = "harena-low-cpu-${var.environment}"
  }
}

# IAM Role for EventBridge to execute SSM Automation
resource "aws_iam_role" "eventbridge" {
  name = "harena-eventbridge-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "harena-eventbridge-role-${var.environment}"
  }
}

# IAM Policy for SSM Automation execution
resource "aws_iam_role_policy" "eventbridge" {
  name = "harena-eventbridge-policy-${var.environment}"
  role = aws_iam_role.eventbridge.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:StartAutomationExecution"
        ]
        Resource = [
          "arn:aws:ssm:${data.aws_region.current.name}:*:automation-definition/AWS-StopEC2Instance:*",
          "arn:aws:ssm:${data.aws_region.current.name}:*:automation-definition/AWS-StartEC2Instance:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:StopInstances",
          "ec2:StartInstances",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      }
    ]
  })
}

# EventBridge Rule - Stop at night (22h Paris = 20h UTC)
resource "aws_cloudwatch_event_rule" "stop_night" {
  count               = var.enable_auto_shutdown && var.shutdown_night ? 1 : 0
  name                = "harena-stop-night-${var.environment}"
  description         = "Stop EC2 at 22h Paris time"
  schedule_expression = "cron(0 20 * * ? *)"

  tags = {
    Name = "harena-stop-night-${var.environment}"
  }
}

resource "aws_cloudwatch_event_target" "stop_night" {
  count     = var.enable_auto_shutdown && var.shutdown_night ? 1 : 0
  rule      = aws_cloudwatch_event_rule.stop_night[0].name
  target_id = "StopEC2Instance"
  arn       = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:automation-definition/AWS-StopEC2Instance"
  role_arn  = aws_iam_role.eventbridge.arn

  input = jsonencode({
    InstanceId = [var.ec2_instance_id]
  })
}

# EventBridge Rule - Start in morning (8h Paris = 6h UTC)
resource "aws_cloudwatch_event_rule" "start_morning" {
  count               = var.enable_auto_shutdown && var.shutdown_night ? 1 : 0
  name                = "harena-start-morning-${var.environment}"
  description         = "Start EC2 at 8h Paris time"
  schedule_expression = "cron(0 6 * * ? *)"

  tags = {
    Name = "harena-start-morning-${var.environment}"
  }
}

resource "aws_cloudwatch_event_target" "start_morning" {
  count     = var.enable_auto_shutdown && var.shutdown_night ? 1 : 0
  rule      = aws_cloudwatch_event_rule.start_morning[0].name
  target_id = "StartEC2Instance"
  arn       = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:automation-definition/AWS-StartEC2Instance"
  role_arn  = aws_iam_role.eventbridge.arn

  input = jsonencode({
    InstanceId = [var.ec2_instance_id]
  })
}

# EventBridge Rule - Stop Friday night (22h Paris)
resource "aws_cloudwatch_event_rule" "stop_weekend" {
  count               = var.enable_auto_shutdown && var.shutdown_weekend ? 1 : 0
  name                = "harena-stop-weekend-${var.environment}"
  description         = "Stop EC2 on Friday night"
  schedule_expression = "cron(0 20 ? * FRI *)"

  tags = {
    Name = "harena-stop-weekend-${var.environment}"
  }
}

resource "aws_cloudwatch_event_target" "stop_weekend" {
  count     = var.enable_auto_shutdown && var.shutdown_weekend ? 1 : 0
  rule      = aws_cloudwatch_event_rule.stop_weekend[0].name
  target_id = "StopEC2Weekend"
  arn       = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:automation-definition/AWS-StopEC2Instance"
  role_arn  = aws_iam_role.eventbridge.arn

  input = jsonencode({
    InstanceId = [var.ec2_instance_id]
  })
}

# EventBridge Rule - Start Monday morning (8h Paris)
resource "aws_cloudwatch_event_rule" "start_monday" {
  count               = var.enable_auto_shutdown && var.shutdown_weekend ? 1 : 0
  name                = "harena-start-monday-${var.environment}"
  description         = "Start EC2 on Monday morning"
  schedule_expression = "cron(0 6 ? * MON *)"

  tags = {
    Name = "harena-start-monday-${var.environment}"
  }
}

resource "aws_cloudwatch_event_target" "start_monday" {
  count     = var.enable_auto_shutdown && var.shutdown_weekend ? 1 : 0
  rule      = aws_cloudwatch_event_rule.start_monday[0].name
  target_id = "StartEC2Monday"
  arn       = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:automation-definition/AWS-StartEC2Instance"
  role_arn  = aws_iam_role.eventbridge.arn

  input = jsonencode({
    InstanceId = [var.ec2_instance_id]
  })
}

# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}
