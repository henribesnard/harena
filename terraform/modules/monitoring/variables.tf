variable "environment" {
  description = "Environment name"
  type        = string
}

variable "enable_auto_shutdown" {
  description = "Enable auto-shutdown"
  type        = bool
}

variable "shutdown_night" {
  description = "Shutdown at night"
  type        = bool
}

variable "shutdown_weekend" {
  description = "Shutdown on weekends"
  type        = bool
}

variable "ec2_instance_id" {
  description = "EC2 instance ID"
  type        = string
}

variable "alert_email" {
  description = "Email for alerts"
  type        = string
}
