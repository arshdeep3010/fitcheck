output "app_url" {
  description = "FitCheck through CloudFront: the page, the API and /how-it-works."
  value       = var.create_function ? "https://${aws_cloudfront_distribution.app[0].domain_name}/" : "not created yet: push the image, then apply with create_function = true"
}

output "function_url" {
  description = "The function's own URL. Answers only CloudFront's signed requests (403 otherwise)."
  value       = var.create_function ? aws_lambda_function_url.app[0].function_url : null
}

output "cloudfront_distribution_id" {
  value = var.create_function ? aws_cloudfront_distribution.app[0].id : null
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
