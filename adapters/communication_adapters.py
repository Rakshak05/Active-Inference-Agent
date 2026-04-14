"""
Communication Adapters — Email (SMTP), webhook, and logging.

Email behaviour:
  - If SMTP_HOST is set in .env → sends a real email via smtplib (TLS).
  - Otherwise → prints a formatted "SIMULATED EMAIL" to the console so you can
    verify the content without needing an SMTP server during development.

Webhook behaviour:
  - Always performs a real HTTP POST to the given URL.
"""

import os
import sys
import json
import smtplib
import urllib.request
import urllib.error
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config


# ── helpers ────────────────────────────────────────────────────────────────────

def _render_template(template: str, record: dict) -> str:
    """Replace {field} placeholders with values from a record dict."""
    try:
        return template.format_map(record)
    except KeyError as e:
        return f"{template}  [missing field: {e}]"


def _smtp_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASS)


def _send_real_email(to: str, subject: str, body: str, html: bool = False) -> str:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = config.SMTP_FROM or config.SMTP_USER
    msg["To"]      = to
    part = MIMEText(body, "html" if html else "plain", "utf-8")
    msg.attach(part)
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(config.SMTP_USER, config.SMTP_PASS)
        smtp.sendmail(msg["From"], [to], msg.as_string())
    return f"Email sent to {to}: '{subject}'"


def _simulate_email(to: str, subject: str, body: str) -> str:
    """Print the email details in lieu of a real send."""
    divider = "─" * 60
    print(f"\n[Email] SIMULATED EMAIL\n{divider}")
    print(f"  To      : {to}")
    print(f"  Subject : {subject}")
    print(f"  Body    :\n{body}")
    print(f"{divider}\n")
    return f"[SIMULATED] Email to {to}: '{subject}'"


# ── adapters ───────────────────────────────────────────────────────────────────

def send_email_adapter(step: dict) -> str:
    """
    Send a single email.
    args: {to: str, subject: str, body: str, html: bool (optional)}
    """
    args    = step.get("args", {})
    to      = str(args.get("to") or args.get("recipient") or args.get("email") or "").strip(",. ")
    subject = str(args.get("subject", "(no subject)"))
    body    = str(args.get("body") or args.get("message") or args.get("content") or "")
    html    = bool(args.get("html", False))

    if not to:
        raise ValueError("send_email: 'to' (recipient) is required.")

    if _smtp_configured():
        return _send_real_email(to, subject, body, html)
    else:
        return _simulate_email(to, subject, body)


def send_emails_bulk_adapter(step: dict) -> str:
    """
    Send one email per record in a list, using template strings.

    args:
      recipients      : list of dicts OR "$varname"
      subject_template: str  (supports {field} placeholders)
      body_template   : str  (supports {field} placeholders)
      to_field        : str  (field in each record containing the email address)
                        defaults to 'email'
      dry_run         : bool (default False) — log only, skip real/simulated send
    """
    args      = step.get("args", {})
    recipients = args.get("recipients", [])
    subj_tmpl  = str(args.get("subject_template", "Reminder"))
    body_tmpl  = str(args.get("body_template", "Hello {name}."))
    to_field   = str(args.get("to_field", "email"))
    dry_run    = bool(args.get("dry_run", False))

    if not isinstance(recipients, list):
        raise TypeError("send_emails_bulk: 'recipients' must be a list of records.")

    results  = []
    sent     = 0
    skipped  = 0
    for record in recipients:
        to = str(record.get(to_field, ""))
        if not to:
            skipped += 1
            results.append({"record": record, "status": "skipped", "reason": f"no '{to_field}' field"})
            continue

        subject = _render_template(subj_tmpl, record)
        body    = _render_template(body_tmpl, record)

        if dry_run:
            print(f"[Email] DRY-RUN → {to}: '{subject}'")
            results.append({"to": to, "subject": subject, "status": "dry_run"})
        elif _smtp_configured():
            try:
                _send_real_email(to, subject, body)
                results.append({"to": to, "subject": subject, "status": "sent"})
                sent += 1
            except Exception as e:
                results.append({"to": to, "error": str(e), "status": "error"})
        else:
            res = _simulate_email(to, subject, body)
            results.append({"to": to, "subject": subject, "status": "simulated", "detail": res})
            sent += 1

    summary = f"Bulk email complete: {sent} sent, {skipped} skipped (total {len(recipients)} recipients)"
    print(f"[Email] {summary}")
    return {"summary": summary, "results": results}


def send_webhook_adapter(step: dict):
    """
    POST a JSON payload to a webhook URL.
    args: {url: str, payload: dict}
    """
    args    = step.get("args", {})
    url     = str(args.get("url", ""))
    payload = args.get("payload") or args.get("body") or args.get("data") or {}
    if not url:
        raise ValueError("send_webhook: 'url' is required.")
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"[Webhook] POST {url} → {resp.status}")
            try:
                return json.loads(body)
            except Exception:
                return body
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Webhook error {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def log_message_adapter(step: dict) -> str:
    """
    Print/log a message. Useful for debugging or notification logging.
    args: {message: str}
    """
    args    = step.get("args", {})
    message = str(args.get("message") or args.get("content") or args.get("text") or "")
    print(f"[Log] {message}")
    return message


def print_table_adapter(step: dict) -> str:
    """
    Pretty-print a list of dicts as a table.
    args: {data: list|"$varname", fields: list (optional)}
    """
    args   = step.get("args", {})
    data   = args.get("data", [])
    fields = args.get("fields")

    if not isinstance(data, list) or not data:
        print("[Table] (empty)")
        return "(empty)"

    if not fields:
        fields = list(data[0].keys()) if isinstance(data[0], dict) else []

    col_widths = {f: max(len(str(f)), max(len(str(r.get(f, ""))) for r in data)) for f in fields}
    header = "  ".join(str(f).ljust(col_widths[f]) for f in fields)
    sep    = "  ".join("-" * col_widths[f] for f in fields)
    lines  = [header, sep]
    for row in data:
        lines.append("  ".join(str(row.get(f, "")).ljust(col_widths[f]) for f in fields))

    table = "\n".join(lines)
    print(f"\n[Table]\n{table}\n")
    return table


# ── registration ───────────────────────────────────────────────────────────────

def setup_communication_adapters(toolgate):
    toolgate.register_adapter("send_email",        send_email_adapter)
    toolgate.register_adapter("send_emails_bulk",  send_emails_bulk_adapter)
    toolgate.register_adapter("send_reminder",     send_email_adapter)   # alias
    toolgate.register_adapter("notify",            send_email_adapter)   # alias
    toolgate.register_adapter("send_webhook",      send_webhook_adapter)
    toolgate.register_adapter("log_message",       log_message_adapter)
    toolgate.register_adapter("print_table",       print_table_adapter)
    print("[Adapters] Communication adapters registered.")
