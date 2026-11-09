# FitCheck on AWS Lambda: the same FastAPI app as `Dockerfile`, on the Lambda base image.
#   docker build -f lambda.Dockerfile -t fitcheck-lambda .
#
# The Lambda Web Adapter turns each Lambda request into plain HTTP for uvicorn and streams
# the response back, so the live step-by-step report (Server-Sent Events) still works.
# Non-HTTP invocations (the EventBridge sync schedule) are posted to /events.
# Outside Lambda the adapter does nothing, so `docker run -p 8000:8000` runs the app as usual.

FROM public.ecr.aws/lambda/python:3.13
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY app ${LAMBDA_TASK_ROOT}/app
COPY profile ${LAMBDA_TASK_ROOT}/profile

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/ \
    AWS_LWA_PASS_THROUGH_PATH=/events

WORKDIR ${LAMBDA_TASK_ROOT}
ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
