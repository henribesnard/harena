variable "environment" {
  description = "Environment name"
  type        = string
}

variable "domain_name" {
  description = "Custom domain name"
  type        = string
  default     = ""
}

variable "backend_public_ip" {
  description = "Public IP address of the backend EC2 instance"
  type        = string
}
