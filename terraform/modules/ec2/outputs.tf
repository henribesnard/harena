output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.backend.id
}

output "public_ip" {
  description = "EC2 public IP"
  value       = aws_instance.backend.public_ip
}

output "backend_security_group_id" {
  description = "Backend security group ID"
  value       = aws_security_group.backend.id
}
