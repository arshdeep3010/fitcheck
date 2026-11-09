variable "aws_region" {
  description = "Region for everything."
  type        = string
  default     = "ap-south-1"
}

variable "name" {
  description = "Name of the function, and prefix for every other resource."
  type        = string
  default     = "fitcheck"
}

variable "create_function" {
  description = "Create the Lambda function. Leave false on the first apply: the image must be in ECR first (see infra/README.md)."
  type        = bool
  default     = false
}

# ---------------------------------------------------------------- GitHub Actions

variable "github_repo" {
  description = "GitHub repository allowed to deploy, as owner/name."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$", var.github_repo))
    error_message = "github_repo must look like owner/name."
  }
}

variable "github_branch" {
  description = "Only pushes to this branch can deploy."
  type        = string
  default     = "main"
}

variable "github_sub_prefix" {
  description = "The repo's OIDC subject prefix when GitHub uses immutable subjects, e.g. repo:owner@123/name@456 (gh api repos/OWNER/NAME/actions/oidc/customization/sub)."
  type        = string
  default     = ""
}

variable "create_github_oidc_provider" {
  description = "Set to false if the account already has the GitHub OIDC provider (token.actions.githubusercontent.com)."
  type        = bool
  default     = true
}

# ---------------------------------------------------------------- the app

variable "github_user" {
  description = "Whose public GitHub repos are the evidence."
  type        = string
}

variable "candidate_name" {
  description = "Shown on the report."
  type        = string
}

variable "scorer" {
  description = "llm_json = the free NVIDIA Nemotron judge. system_one = Mercury Decide on OpenRouter (paid; falls back to Nemotron when the credits run out)."
  type        = string
  default     = "llm_json"

  validation {
    condition     = contains(["llm_json", "system_one"], var.scorer)
    error_message = "scorer must be llm_json or system_one."
  }
}

variable "sync_minutes" {
  description = "How often the schedule checks GitHub for new commits (0 = off). A quiet check is a few small GitHub API calls and no embeddings."
  type        = number
  default     = 30
}

variable "rate_limit_per_hour" {
  description = "Checks per visitor per hour."
  type        = number
  default     = 20
}

variable "daily_limit" {
  description = "Checks per UTC day across all visitors (protects the free model quotas)."
  type        = number
  default     = 200
}

variable "memory_mb" {
  description = "Lambda memory. The app mostly waits on model APIs; 1024 MB starts fast enough."
  type        = number
  default     = 1024
}

variable "log_retention_days" {
  type    = number
  default = 14
}

# ---------------------------------------------------------------- cost guard

variable "budget_email" {
  description = "Email for a monthly AWS budget alert (empty = no budget)."
  type        = string
  default     = ""
}

variable "budget_usd" {
  description = "Monthly budget in USD; you get an email at 80% of it."
  type        = number
  default     = 2
}
