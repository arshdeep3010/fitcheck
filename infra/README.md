# FitCheck on AWS Lambda

One Lambda function runs the same FastAPI app as `docker compose`, behind **CloudFront** and a
**function URL** that streams the step-by-step report and accepts only CloudFront's signed
requests. Terraform creates it; GitHub Actions deploys every push to `main` that passes the tests.

```
visitor ──HTTPS──> CloudFront ──signed (OAC)──> function URL (AWS_IAM, streaming)
                                                  └──> Lambda: Lambda Web Adapter -> uvicorn -> FastAPI
EventBridge, every 30 min ──{"action":"sync"}──> same function -> POST /events -> GitHub sync
                                                     │
                       Neon Postgres + pgvector  <───┘  index, shared rate limits, sync status
                       SSM Parameter Store       <───   API keys, database URL, sync token
```

| Resource | Why |
|---|---|
| ECR repository `fitcheck` | The image; keeps the last 3 |
| Lambda `fitcheck` (1024 MB, 300 s, x86_64) | The app. No VPC: it only calls public APIs, and a VPC would need a NAT gateway (~$30+/month) |
| CloudFront distribution + origin access control | The public address. HTTPS at the edge; caches only the two pages (5 min, cleared on every deploy); `/api/*` is never cached or compressed, so the report streams. Signs every request to the function |
| Function URL, auth `AWS_IAM`, `RESPONSE_STREAM` | Answers only the distribution's signed requests (403 to anyone else). The visitor's IP comes from `CloudFront-Viewer-Address`. Abuse limits are in the app: 20 checks per visitor per hour, 200 per day overall, kept in Postgres so every instance shares them |
| EventBridge rule, `rate(30 minutes)` | A frozen function can't wake itself, so the schedule triggers the GitHub sync. The event carries a token, because `/events` is reachable through CloudFront too |
| SSM `/fitcheck/*` (SecureString) | `nvidia_api_key`, `database_url`, `openrouter_api_key`, `github_token`: you set them; `sync_token`: generated |
| IAM role `fitcheck` | Logs, and reading its own `/fitcheck/*` parameters. Nothing else |
| IAM role `fitcheck-deploy` (GitHub OIDC) | Push to this ECR repository; update, invoke and read the URL of this one function; read the sync token; clear this distribution's cache. No AWS keys in GitHub |
| SSM `/fitcheck-deploy/cloudfront` (String) | The distribution's id and URL, for the deploy workflow |
| CloudWatch log group, 14 days | Logs |
| Budget (optional) | Email at 80% of `budget_usd` a month |

## Cost

Expected: **₹0 to a few rupees a month**. Lambda's always-free allowance (1M requests, 400,000 GB-seconds) covers ~20,000 checks a month at 1 GB, and the daily limit stops at ~6,000. CloudFront's always-free tier (1 TB out and 10M requests a month), function URLs, EventBridge rules, SSM standard parameters and this much logging are free. ECR storage is a few cents. Neon's free plan holds the index (a few MB). The models are on NVIDIA's free tier; Mercury Decide (`scorer = "system_one"`) is ~₹0.03 a check. This is from AWS pricing, not a measured bill: set `budget_email` to find out.

## First deploy

You need: the AWS CLI signed in, Terraform 1.6+ (or Docker: see below), a GitHub repo with this code, and a **Neon** project (free) with its **direct** connection string (not the pooler: the app keeps session settings).

1. **Settings.** `cp terraform.tfvars.example terraform.tfvars`, then fill in `github_repo`, `github_user`, `candidate_name`. If the account already has the GitHub OIDC provider (it does if you deployed AgentOps or the portfolio), set `create_github_oidc_provider = false`. If GitHub uses immutable subjects for your repos, set `github_sub_prefix` from:
   ```bash
   gh api repos/OWNER/fitcheck/actions/oidc/customization/sub
   ```
2. **First apply** (`create_function = false`): the image repository, roles, secrets placeholders.
   ```bash
   terraform init
   terraform apply
   ```
3. **Secrets.** Put the real values in SSM (Terraform never sees them):
   ```bash
   aws ssm put-parameter --overwrite --type SecureString --name /fitcheck/nvidia_api_key --value "nvapi-..."
   aws ssm put-parameter --overwrite --type SecureString --name /fitcheck/database_url  --value "postgresql://..."
   ```
   `openrouter_api_key` only for `scorer = "system_one"`; `github_token` is optional.
4. **GitHub variables.** `terraform output github_variables`, then set `AWS_ROLE_ARN` and `AWS_REGION` in the repo (Settings → Secrets and variables → Actions → Variables).
5. **Push the image.** Push to `main` (or Actions → deploy → Run workflow). The first run only pushes the image, because the function doesn't exist yet.
6. **Create the function.** Set `create_function = true` in `terraform.tfvars`, then `terraform apply -parallelism=1` (one change to the function's permissions at a time; CloudFront takes about 5 minutes). `terraform output app_url` is the live (CloudFront) address.
7. **Fill the index.** Run the deploy workflow once more (Actions → deploy → Run workflow). It syncs with GitHub straight away, then checks `/health`. From then on every push to `main` deploys, and the schedule keeps the index fresh.

Terraform through Docker, with your AWS sign-in mounted:
```bash
docker run --rm -it -v "$PWD:/w" -w /w -v "$HOME/.aws:/root/.aws:ro" hashicorp/terraform:1.9.8 plan
```

## Day to day

- **New code:** push to `main`. ci runs the tests (with a real pgvector), then deploy ships it.
- **New project on GitHub:** nothing to do; it's citable within 30 minutes.
- **CV change** (`profile/profile.md`): push to `main`; the deploy re-indexes it.
- **Sync now:** run the deploy workflow (Actions → deploy → Run workflow). It invokes the function directly with the sync token; CloudFront would give up on a long sync after 60 s.
- **Logs:** `aws logs tail /aws/lambda/fitcheck --follow`
- **Turn it off:** `terraform destroy` (the Neon database is separate).

State is local (`infra/terraform.tfstate`, git-ignored). It holds the generated sync token: back it up and never commit it.
