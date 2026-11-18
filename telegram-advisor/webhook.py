#!/usr/bin/env python3
"""
Webhook transport for the advisor bot.

Telegram POSTs each update to a public HTTPS URL instead of the bot polling
for it. This file is the HTTP endpoint; `bot.Bot.handle()` does the rest.

    python3 webhook.py                      serve on $PORT (default 8080)
    python3 webhook.py --set https://host   register the webhook with Telegram
    python3 webhook.py --info               show what Telegram has registered
    python3 webhook.py --delete             unregister (needed before polling)
    python3 webhook.py --new-secret         print a fresh secret token

Extra configuration, on top of bot.py's:
  TELEGRAM_WEBHOOK_SECRET  required -- Telegram echoes it in a header on
                           every request; anything without it is rejected
  PUBLIC_URL               optional -- https://host of this deployment; if
                           set, the webhook is registered at startup
  PORT                     optional -- default 8080
  WEBHOOK_PATH             optional -- default /telegram/webhook
  WEBHOOK_SYNC             optional -- "1" to finish handling an update
                           before answering Telegram (for hosts that freeze
                           the process once the response is sent)

TLS is not handled here. Telegram only delivers to HTTPS, so run this behind
whatever terminates TLS on your host -- every PaaS does it for you.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bot import Bot, configure  # noqa: E402
from tg import TelegramError  # noqa: E402

log = logging.getLogger("advisor.webhook")

MAX_BODY = 1024 * 1024        # Telegram updates are a few KB; refuse anything silly
REMEMBERED_UPDATES = 2000     # how many update_ids to remember for de-duplication
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"
SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{16,256}$")   # Telegram's charset; 16+ is ours


class WebhookApp:
    """Authenticates nothing itself -- the handler does -- but owns
    de-duplication, per-chat ordering and the worker pool."""

    def __init__(self, bot: Bot, secret: str, path: str, sync: bool = False) -> None:
        self.bot = bot
        self.secret = secret
        self.path = path
        self.sync = sync
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="update")
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._seen_lock = threading.Lock()
        self._chat_locks: dict[int, threading.Lock] = {}
        self._chat_locks_guard = threading.Lock()

    def _first_time(self, update: dict) -> bool:
        """Telegram re-delivers an update until it gets a 2xx, so the same
        update_id can arrive twice. Answer it once."""
        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            return True
        with self._seen_lock:
            if update_id in self._seen:
                return False
            self._seen[update_id] = None
            while len(self._seen) > REMEMBERED_UPDATES:
                self._seen.popitem(last=False)
            return True

    def _chat_lock(self, update: dict) -> threading.Lock:
        chat_id = ((update.get("message") or {}).get("chat") or {}).get("id", 0)
        with self._chat_locks_guard:
            return self._chat_locks.setdefault(chat_id, threading.Lock())

    def _handle(self, update: dict) -> None:
        # One update at a time per chat: the why-tree is a small state
        # machine and two quick messages must not interleave.
        with self._chat_lock(update):
            try:
                self.bot.handle(update)
            except Exception:
                log.exception("failed handling update %s", update.get("update_id"))

    def dispatch(self, update: dict) -> None:
        if not self._first_time(update):
            log.info("duplicate update %s ignored", update.get("update_id"))
            return
        if self.sync:
            self._handle(update)
        else:
            self._pool.submit(self._handle, update)

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def make_handler(app: WebhookApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "advisor"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the access log out of stderr
            log.debug("%s %s", self.address_string(), fmt % args)

        def _reply(self, status: int, body: str = "", close: bool = False) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if close:
                # The request body was not read, so the connection cannot be
                # reused: the leftover bytes would be parsed as a request.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/healthz":
                self._reply(200, "ok")
            else:
                self._reply(404)

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != app.path:
                self._reply(404, close=True)
                return

            # Check the secret before reading a byte of the body. Without
            # this, anyone who learns the URL could forge an update "from"
            # an allowed user id and drive the bot.
            supplied = self.headers.get(SECRET_HEADER, "")
            if not hmac.compare_digest(supplied.encode(), app.secret.encode()):
                log.warning("rejected request with a bad secret from %s", self.address_string())
                self._reply(403, close=True)
                return

            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._reply(411, close=True)
                return
            if length <= 0:
                self._reply(400, close=True)
                return
            if length > MAX_BODY:
                self._reply(413, close=True)
                return

            try:
                update = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reply(400)
                return
            if not isinstance(update, dict):
                self._reply(400)
                return

            app.dispatch(update)
            # 200 as soon as the update is accepted: Telegram re-delivers
            # on a slow answer, and a model call can take a while.
            self._reply(200, "ok")

    return Handler


# --------------------------------------------------------------------------
# Config + commands
# --------------------------------------------------------------------------


def webhook_settings() -> tuple[str, str, int, str, bool]:
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    path = os.environ.get("WEBHOOK_PATH", "/telegram/webhook").strip() or "/telegram/webhook"
    if not path.startswith("/"):
        path = "/" + path
    port = int(os.environ.get("PORT", "8080"))
    public_url = os.environ.get("PUBLIC_URL", "").strip().rstrip("/")
    sync = os.environ.get("WEBHOOK_SYNC", "").strip().lower() in ("1", "true", "yes")
    return secret, path, port, public_url, sync


def check_secret(secret: str) -> str | None:
    if not secret:
        return (
            "TELEGRAM_WEBHOOK_SECRET ยังไม่ได้ตั้งค่า — สร้างด้วย "
            "`python3 webhook.py --new-secret` แล้วใส่ใน .env"
        )
    if not SECRET_RE.match(secret):
        return "TELEGRAM_WEBHOOK_SECRET ต้องยาว 16–256 ตัว และใช้ได้เฉพาะ A-Z a-z 0-9 _ -"
    return None


def register(bot: Bot, base_url: str, path: str, secret: str) -> bool:
    if not base_url.startswith("https://"):
        print("error: Telegram ส่ง webhook ได้เฉพาะ URL ที่เป็น https://", file=sys.stderr)
        return False
    url = base_url.rstrip("/") + path
    try:
        bot.tg.set_webhook(url, secret)
    except (TelegramError, OSError) as exc:
        print(f"error: ลงทะเบียน webhook ไม่สำเร็จ — {exc}", file=sys.stderr)
        return False
    print(f"webhook registered: {url}")
    return True


def serve(bot: Bot, secret: str, path: str, port: int, public_url: str, sync: bool) -> int:
    app = WebhookApp(bot, secret, path, sync=sync)
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(app))
    server.daemon_threads = True

    def shutdown(*_args) -> None:
        # shutdown() blocks until serve_forever returns, so it must not run
        # on the thread that is inside serve_forever.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(
        "listening on :%d%s | %d models | mode=%s | style=%s | %s",
        port,
        path,
        len(bot.bundle.models),
        "llm" if bot.advisor.available else "offline",
        bot.config.style,
        "sync" if sync else "async",
    )
    if public_url:
        # A failed registration is logged, not fatal: the health check
        # should still pass so the host does not restart-loop the service.
        register(bot, public_url, path, secret)
    else:
        log.info("PUBLIC_URL not set — register with: python3 webhook.py --set https://<host>")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Webhook transport for the advisor bot.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--set", metavar="URL", help="register https://host with Telegram")
    group.add_argument("--info", action="store_true", help="show the registered webhook")
    group.add_argument("--delete", action="store_true", help="unregister the webhook")
    group.add_argument("--new-secret", action="store_true", help="print a fresh secret")
    args = parser.parse_args()

    if args.new_secret:
        print(secrets.token_urlsafe(32))
        return 0

    config = configure()
    if config is None:
        return 1
    secret, path, port, public_url, sync = webhook_settings()
    bot = Bot(config)

    try:
        if args.info:
            info = bot.tg.webhook_info()
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
        if args.delete:
            bot.tg.delete_webhook()
            print("webhook deleted")
            return 0
    except (TelegramError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    problem = check_secret(secret)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    if args.set:
        return 0 if register(bot, args.set, path, secret) else 1

    return serve(bot, secret, path, port, public_url, sync)


if __name__ == "__main__":
    raise SystemExit(main())
