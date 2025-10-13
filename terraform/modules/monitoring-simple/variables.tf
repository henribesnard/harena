variable "environment" {
  description = "Environment name"
  type        = string
}

variable "enable_auto_shutdown" {
  description = "Enable auto-shutdown for EC2"
  type        = bool
  default     = true
}

variable "shutdown_night" {
  description = "Shutdown EC2 at night"
  type        = bool
  default     = true
}

variable "shutdown_weekend" {
  description = "Shutdown EC2 on weekends"
  type        = bool
  default     = true
}

variable "ec2_instance_id" {
  description = "EC2 instance ID to monitor"
  type        = string
}

variable "alert_email" {
  description = "Email for alerts (optional)"
  type        = string
  default     = ""
}
