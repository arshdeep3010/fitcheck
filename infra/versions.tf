terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # State is kept locally in infra/terraform.tfstate (git-ignored). Back it up, or move it to
  # an S3 backend: losing it means Terraform no longer knows about the resources it created.
  # It also holds the generated sync token, so never commit it.
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = var.name
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "me" {}
data "aws_region" "current" {}

locals {
  account    = data.aws_caller_identity.me.account_id
  region     = data.aws_region.current.region
  fn         = var.create_function ? 1 : 0
  fn_name    = var.name # fixed, so IAM policies can name it before it exists
  fn_arn     = "arn:aws:lambda:${local.region}:${local.account}:function:${local.fn_name}"
  ssm_prefix = "/${var.name}"
}
