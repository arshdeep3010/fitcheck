# CloudFront in front of the function. Visitors get HTTPS at an edge location near them, the
# two pages are cached there for a few minutes, and the function URL only accepts requests
# CloudFront has signed (AWS_IAM + origin access control), so nobody can go around it.
#
# Two things the app does for this:
# - POSTs carry x-amz-content-sha256 (the SHA-256 of the body), which CloudFront needs to
#   sign them; the page computes it (app/static/index.html, bodyHash).
# - The visitor's IP comes from the CloudFront-Viewer-Address header (BEHIND_CLOUDFRONT=true);
#   the connection's own address is now always a CloudFront server.
#
# Cost: CloudFront's always-free tier (1 TB out, 10M requests a month) covers this app.

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

# All viewer headers, cookies and query strings except Host (a function URL must see its own
# host name), plus CloudFront's viewer headers such as CloudFront-Viewer-Address.
data "aws_cloudfront_origin_request_policy" "all_but_host" {
  name = "Managed-AllViewerExceptHostHeader"
}

# Cache only what the app marks cacheable: the pages send "s-maxage=300". Anything without a
# Cache-Control header (every API response) gets the default TTL of 0, so it isn't cached.
resource "aws_cloudfront_cache_policy" "pages" {
  name        = "${var.name}-pages"
  comment     = "Respect the app's s-maxage; cache nothing else"
  min_ttl     = 0
  default_ttl = 0
  max_ttl     = 3600

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_gzip   = true
    enable_accept_encoding_brotli = true
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "none"
    }
    query_strings_config {
      query_string_behavior = "none"
    }
  }
}

resource "aws_cloudfront_origin_access_control" "fn" {
  count                             = local.fn
  name                              = "${var.name}-lambda"
  description                       = "CloudFront signs every request to the FitCheck function URL"
  origin_access_control_origin_type = "lambda"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

locals {
  fn_url_host = var.create_function ? trimsuffix(trimprefix(aws_lambda_function_url.app[0].function_url, "https://"), "/") : ""
}

resource "aws_cloudfront_distribution" "app" {
  count           = local.fn
  enabled         = true
  comment         = var.name
  is_ipv6_enabled = true
  http_version    = "http2and3"
  # Edge locations in North America, Europe, Asia (India included), the Middle East and
  # Africa. PriceClass_100 would serve Indian visitors from Europe.
  price_class = "PriceClass_200"

  origin {
    origin_id                = "lambda"
    domain_name              = local.fn_url_host
    origin_access_control_id = aws_cloudfront_origin_access_control.fn[0].id

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      # The longest quiet gap in a streamed check is one model call (~15 s); 60 s is the
      # most CloudFront allows without a quota increase. Syncs don't come this way.
      origin_read_timeout      = 60
      origin_keepalive_timeout = 5
    }
  }

  # The pages, /health and the rest: cached only when the app says so.
  default_cache_behavior {
    target_origin_id         = "lambda"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = aws_cloudfront_cache_policy.pages.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_but_host.id
    compress                 = true
  }

  # The API, including the streamed report: never cached, never compressed (compression
  # would hold back the Server-Sent Events until a buffer fills).
  ordered_cache_behavior {
    path_pattern             = "/api/*"
    target_origin_id         = "lambda"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_but_host.id
    compress                 = false
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# Only this distribution may call the function (both actions are needed for function URLs).
resource "aws_lambda_permission" "cloudfront_url" {
  count         = local.fn
  statement_id  = "AllowCloudFrontServicePrincipal"
  action        = "lambda:InvokeFunctionUrl"
  function_name = aws_lambda_function.app[0].function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.app[0].arn
}

resource "aws_lambda_permission" "cloudfront_invoke" {
  count         = local.fn
  statement_id  = "AllowCloudFrontServicePrincipalInvokeFunction"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app[0].function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.app[0].arn
  # One change to the function's policy at a time (409 "concurrent update" otherwise).
  depends_on = [aws_lambda_permission.cloudfront_url]
}

# Where the deploy workflow finds the distribution (to clear the page cache and check the live
# app). Outside the app's /fitcheck/ prefix, so the function doesn't load it.
resource "aws_ssm_parameter" "cloudfront" {
  count       = local.fn
  name        = "/${var.name}-deploy/cloudfront"
  description = "CloudFront distribution id and URL, for the deploy workflow"
  type        = "String"
  value = jsonencode({
    id  = aws_cloudfront_distribution.app[0].id
    url = "https://${aws_cloudfront_distribution.app[0].domain_name}/"
  })
}
