output "app_url" {
  description = "FitCheck: the page, the API and /how-it-works."
  value       = var.create_function ? aws_lambda_function_url.app[0].function_url : "not created yet: push the image, then apply with create_function = true"
}

output "github_variables" {
  description = "Set these as repository variables (Settings > Secrets and variables > Actions > Variables)."
  value = {
    AWS_ROLE_ARN = aws_iam_role.deploy.arn
    AWS_REGION   = local.region
  }
}

output "ecr_repository" {
  value = aws_ecr_repository.app.repository_url
}

output "secrets_to_set" {
  description = "SSM parameters to fill in yourself (Terraform never sees the values)."
  value       = [for p in aws_ssm_parameter.secret : p.name]
}
