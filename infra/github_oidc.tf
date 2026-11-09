# GitHub Actions signs in to AWS with a short-lived OIDC token: no AWS keys are stored in GitHub.

locals {
  github_oidc_url = "token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_github_oidc_provider ? 1 : 0
  url            = "https://${local.github_oidc_url}"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://${local.github_oidc_url}"
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

# Only pushes to one branch of one repository can take this role.
data "aws_iam_policy_document" "deploy_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.github_oidc_url}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.github_oidc_url}:sub"
      # Older repos send repo:owner/name; newer ones send repo:owner@ID/name@ID.
      values = compact([
        "repo:${var.github_repo}:ref:refs/heads/${var.github_branch}",
        var.github_sub_prefix == "" ? "" : "${var.github_sub_prefix}:ref:refs/heads/${var.github_branch}",
      ])
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.name}-deploy"
  description          = "Assumed by GitHub Actions to push the FitCheck image and update the function"
  assume_role_policy   = data.aws_iam_policy_document.deploy_trust.json
  max_session_duration = 3600
}

# Least privilege: push to one repository, update and invoke one function, read the sync token
# (the workflow syncs right after a deploy, so a fresh database is filled at once).
data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action has no resource-level permissions
  }

  statement {
    sid = "PushImage"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
      "ecr:DescribeImages", "ecr:GetDownloadUrlForLayer", "ecr:InitiateLayerUpload",
      "ecr:PutImage", "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.app.arn]
  }

  statement {
    sid = "UpdateAndSyncFunction"
    actions = [
      "lambda:UpdateFunctionCode", "lambda:GetFunction", "lambda:GetFunctionConfiguration",
      "lambda:GetFunctionUrlConfig", "lambda:InvokeFunction",
    ]
    resources = [local.fn_arn]
  }

  statement {
    sid       = "ReadSyncToken"
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.sync_token.arn]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy-${var.name}"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
