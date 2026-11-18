# Build from the repository root, so the skill bundle is in the context:
#   docker build -f telegram-advisor/Dockerfile -t mm-advisor .
FROM python:3.12-slim

WORKDIR /app
# The bot reads the bundle from ../skills/mental-models relative to itself.
COPY skills/mental-models /app/skills/mental-models
COPY telegram-advisor /app/telegram-advisor
RUN pip install --no-cache-dir -r /app/telegram-advisor/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    ADVISOR_STATE_FILE=/tmp/advisor/conversations.json
EXPOSE 8080
USER nobody

CMD ["python", "/app/telegram-advisor/webhook.py"]
