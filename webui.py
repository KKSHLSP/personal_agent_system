"""Tiny Web UI and HTTP channel for the personal agent system.

Run with:
    python3 webui.py

Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import urllib.parse
import urllib.request
import urllib.error

from agent import AgentResult, IncomingMessage, build_demo_system

MAX_BODY_BYTES = 65_536
MAX_CONTENT_LEN = 4_096
MAX_ID_LEN = 128
_ALLOWED_URL_SCHEMES = ("http://", "https://")

_SERVER_START_TIME = datetime.now(timezone.utc)


class _BodyTooLargeError(Exception):
    pass


HTML = """<!doctype html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal Agent WebUI</title>
  <style>
    :root { color-scheme: light; --bg: #f5f7fb; --panel: #ffffff; --ink: #1d2733; --muted: #637083; --line: #d9e0ea; --accent: #1f7a8c; --danger: #b42318; --ok: #247a3b; --warn: #9a5b00; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    main { width: min(1080px, calc(100vw - 32px)); margin: 28px auto; display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 18px; align-items: start; }
    section, form { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; box-shadow: 0 8px 28px rgb(34 48 74 / 7%); }
    h1 { grid-column: 1 / -1; margin: 0 0 2px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 17px; }
    label { display: grid; gap: 6px; margin-bottom: 12px; color: var(--muted); }
    select, input, textarea, button { width: 100%; border-radius: 6px; border: 1px solid var(--line); font: inherit; color: var(--ink); background: #fff; }
    select, input { height: 40px; padding: 0 10px; }
    textarea { min-height: 110px; resize: vertical; padding: 10px; }
    button { height: 42px; border-color: var(--accent); background: var(--accent); color: #fff; cursor: pointer; font-weight: 650; }
    button:disabled { opacity: .65; cursor: wait; }
    .hint { margin: -4px 0 14px; color: var(--muted); }
    .result { min-height: 352px; display: grid; gap: 12px; align-content: start; }
    .badge { display: inline-flex; width: fit-content; padding: 3px 9px; border-radius: 999px; font-size: 13px; font-weight: 700; background: #eef4f7; color: var(--accent); }
    .badge.AUTO_REPLY { color: var(--ok); background: #edf8f0; }
    .badge.SUGGEST_ONLY, .badge.ASK_USER { color: var(--warn); background: #fff6e6; }
    .badge.REJECT { color: var(--danger); background: #fff0ee; }
    .reply { padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; white-space: pre-wrap; }
    dl { margin: 0; display: grid; gap: 10px; }
    dt { color: var(--muted); font-size: 13px; }
    dd { margin: 2px 0 0; }
    ul { margin: 4px 0 0; padding-left: 18px; }
    pre { margin: 0; padding: 12px; overflow: auto; background: #101820; color: #e7edf5; border-radius: 8px; font-size: 13px; }
    .stack { display: grid; gap: 18px; }
    .audit-list {
      display: grid;
      gap: 8px;
      max-height: 240px;
      overflow: auto;
      padding-right: 4px;
    }
    .audit-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }
    .audit-meta {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>个人数字分身 Agent</h1>
    <div class="stack">
      <form id="message-form">
        <h2>模拟聊天渠道</h2>
        <p class="hint">选择联系人并发送一条消息，系统会按权限、知识库、风格和安全策略生成结果。</p>
        <label>联系人
          <select name="sender_id">
            <option value="classmate_a">classmate_a：允许自动回复，口语简短</option>
            <option value="classmate_b">classmate_b：只生成建议，正式礼貌</option>
            <option value="unknown">unknown：默认权限</option>
          </select>
        </label>
        <label>会话 ID
          <input name="conversation_id" value="web-conv-a">
        </label>
        <label>消息内容
          <textarea name="content">资料在哪里？</textarea>
        </label>
        <button type="submit">发送到 Agent</button>
      </form>
      <form id="ai-form">
        <h2>本地 AI 渠道测试</h2>
        <p class="hint">兼容 OpenAI `/v1/chat/completions`，用于测试 8001 上的本地 oMLX 服务。</p>
        <label>Base URL
          <input name="base_url" value="http://127.0.0.1:8001">
        </label>
        <label>Model
          <input name="model" value="Qwen3.6-27B-4bit">
        </label>
        <label>API Key
          <input name="api_key" type="password" placeholder="留空时读取 LOCAL_AI_API_KEY">
        </label>
        <label>Prompt
          <textarea name="prompt">请用一句话介绍你自己。</textarea>
        </label>
        <button type="submit">测试本地 AI</button>
      </form>
    </div>
    <section class="result">
      <h2>处理结果</h2>
      <span id="action" class="badge">WAITING</span>
      <div id="reply" class="reply">尚未发送消息。</div>
      <dl>
        <div><dt>策略说明</dt><dd id="reason">-</dd></div>
        <div><dt>可信度</dt><dd id="confidence">-</dd></div>
        <div><dt>依据</dt><dd><ul id="evidence"></ul></dd></div>
        <div><dt>安全标记</dt><dd><ul id="flags"></ul></dd></div>
      </dl>
      <h2>Raw JSON</h2>
      <pre id="raw">{}</pre>
      <h2>Local AI</h2>
      <pre id="ai-raw">{}</pre>
      <h2>审计记录</h2>
      <button id="refresh-audit" type="button">刷新审计记录</button>
      <div id="audit-list" class="audit-list">暂无记录。</div>
    </section>
  </main>
  <script>
    const form = document.querySelector("#message-form");
    const action = document.querySelector("#action");
    const reply = document.querySelector("#reply");
    const reason = document.querySelector("#reason");
    const confidence = document.querySelector("#confidence");
    const evidence = document.querySelector("#evidence");
    const flags = document.querySelector("#flags");
    const raw = document.querySelector("#raw");
    const aiForm = document.querySelector("#ai-form");
    const aiRaw = document.querySelector("#ai-raw");
    const auditButton = document.querySelector("#refresh-audit");
    const auditList = document.querySelector("#audit-list");

    function fillList(node, values) {
      node.innerHTML = "";
      if (!values.length) {
        const li = document.createElement("li");
        li.textContent = "-";
        node.appendChild(li);
        return;
      }
      for (const value of values) {
        const li = document.createElement("li");
        li.textContent = value;
        node.appendChild(li);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = form.querySelector("button");
      button.disabled = true;
      const payload = Object.fromEntries(new FormData(form).entries());
      try {
        const response = await fetch("/api/message", { method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        action.className = "badge " + data.decision.action;
        action.textContent = data.decision.action;
        reply.textContent = data.draft.text;
        reason.textContent = data.decision.reason;
        confidence.textContent = data.decision.confidence.toFixed(2);
        fillList(evidence, data.draft.evidence);
        fillList(flags, data.draft.safety_flags);
        raw.textContent = JSON.stringify(data, null, 2);
        loadAudit();
      } catch (error) {
        action.className = "badge REJECT";
        action.textContent = "ERROR";
        reply.textContent = String(error);
      } finally {
        button.disabled = false;
      }
    });

    aiForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = aiForm.querySelector("button");
      button.disabled = true;
      const payload = Object.fromEntries(new FormData(aiForm).entries());
      try {
        const response = await fetch("/api/local-ai-test", { method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload) });
        const data = await response.json();
        aiRaw.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        aiRaw.textContent = String(error);
      } finally {
        button.disabled = false;
      }
    });

    async function loadAudit() {
      try {
        const response = await fetch("/api/audit");
        const data = await response.json();
        const entries = Array.isArray(data) ? data : data.entries || [];
        if (!entries.length) {
          auditList.textContent = "暂无记录。";
          return;
        }
        auditList.innerHTML = "";
        for (const entry of entries.slice().reverse()) {
          const item = document.createElement("div");
          item.className = "audit-item";
          const title = document.createElement("strong");
          title.textContent = `${entry.action} · ${entry.sender_id}`;
          const meta = document.createElement("div");
          meta.className = "audit-meta";
          meta.textContent = `${entry.conversation_id} · ${new Date(entry.timestamp).toLocaleString()} · confidence ${Number(entry.confidence).toFixed(2)}`;
          const reasonText = document.createElement("div");
          reasonText.textContent = entry.reason;
          item.append(title, meta, reasonText);
          auditList.appendChild(item);
        }
      } catch (error) {
        auditList.textContent = String(error);
      }
    }

    auditButton.addEventListener("click", loadAudit);
    loadAudit();
  </script>
</body>
</html>
"""


def result_to_dict(result: AgentResult) -> dict[str, object]:
    return {
        "message": {
            "sender_id": result.message.sender_id,
            "conversation_id": result.message.conversation_id,
            "content": result.message.content,
            "timestamp": result.message.timestamp.isoformat(),
            "platform": result.message.platform,
        },
        "decision": {
            "action": result.decision.action.value,
            "reason": result.decision.reason,
            "confidence": result.decision.confidence,
            "required_user_confirmation": result.decision.required_user_confirmation,
        },
        "draft": {
            "text": result.draft.text,
            "evidence": result.draft.evidence,
            "style_notes": result.draft.style_notes,
            "safety_flags": result.draft.safety_flags,
        },
        "should_send": result.should_send,
    }


def call_local_ai(base_url: str, model: str, prompt: str, api_key: str = "", timeout: float = 60) -> dict[str, object]:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是个人数字分身系统的本地模型测试助手，回答要简洁。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 160,
        "stream": False,
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"content-type": "application/json"}
    api_key = api_key or os.environ.get("LOCAL_AI_API_KEY", "")
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8") or "{}")
            return {"ok": True, "status": response.status, "endpoint": endpoint, "model": model, "content": clean_model_text(_extract_chat_content(data)), "raw": data}
    except urllib.error.HTTPError as error:
        return {"ok": False, "status": error.code, "endpoint": endpoint, "error": error.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as error:
        return {"ok": False, "status": 0, "endpoint": endpoint, "error": str(error.reason)}


def list_local_ai_models(base_url: str, api_key: str = "", timeout: float = 15) -> dict[str, object]:
    endpoint = base_url.rstrip("/") + "/v1/models"
    headers = {}
    api_key = api_key or os.environ.get("LOCAL_AI_API_KEY", "")
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8") or "{}")
            models = [item["id"] for item in data.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
            return {"ok": True, "status": response.status, "endpoint": endpoint, "models": models, "raw": data}
    except urllib.error.HTTPError as error:
        return {"ok": False, "status": error.code, "endpoint": endpoint, "error": error.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as error:
        return {"ok": False, "status": 0, "endpoint": endpoint, "error": str(error.reason)}


def clean_model_text(text: str) -> str:
    cleaned = text.strip()
    thinking_markers = ("Here's a thinking process:", "Thinking process:", "<think>")
    if not cleaned.startswith(thinking_markers):
        return cleaned
    final_markers = ("Final answer:", "最终答案：", "</think>")
    for marker in final_markers:
        if marker in cleaned:
            return cleaned.split(marker, 1)[1].strip()
    return cleaned


def _extract_chat_content(data: dict[str, object]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


class WebAgentHandler(BaseHTTPRequestHandler):
    system = build_demo_system()

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send_text(HTTPStatus.OK, HTML, "text/html; charset=utf-8")
            return
        _audit_parsed = urllib.parse.urlparse(self.path)
        if _audit_parsed.path == "/api/audit":
            _aq = urllib.parse.parse_qs(_audit_parsed.query)
            try:
                _offset = max(0, int(_aq.get("offset", ["0"])[0]))
            except ValueError:
                _offset = 0
            try:
                _limit = max(0, int(_aq.get("limit", ["0"])[0]))
            except ValueError:
                _limit = 0
            self._send_json(HTTPStatus.OK, self.system.audit_log.to_page(offset=_offset, limit=_limit))
            return
        if self.path == "/api/health":
            now = datetime.now(timezone.utc)
            stats = self.system.audit_log.stats()
            self._send_json(HTTPStatus.OK, {
                "status": "ok",
                "started_at": _SERVER_START_TIME.isoformat(),
                "uptime_seconds": (now - _SERVER_START_TIME).total_seconds(),
                "total_processed": stats.total,
            })
            return
        if self.path == "/api/stats":
            s = self.system.audit_log.stats()
            rl = self.system.rate_limiter
            self._send_json(HTTPStatus.OK, {
                "total": s.total,
                "by_action": s.by_action,
                "auto_sent_count": s.auto_sent_count,
                "flagged_count": s.flagged_count,
                "mean_confidence": s.mean_confidence,
                "rate_limited_count": rl.total_rejected() if rl is not None else 0,
            })
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/local-ai-models":
            query = urllib.parse.parse_qs(parsed.query)
            base_url = query.get("base_url", ["http://127.0.0.1:8001"])[0]
            api_key = query.get("api_key", [""])[0]
            if not any(base_url.startswith(s) for s in _ALLOWED_URL_SCHEMES):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "base_url must start with http:// or https://"})
                return
            result = list_local_ai_models(base_url, api_key)
            status = HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY
            self._send_json(status, result)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/message":
            if parsed.path == "/api/local-ai-test":
                self._handle_local_ai_test()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            payload = self._read_json()
            content = str(payload.get("content", "")).strip()
            sender_id = str(payload.get("sender_id", "unknown")).strip() or "unknown"
            conversation_id = str(payload.get("conversation_id", f"web-{sender_id}")).strip() or f"web-{sender_id}"
            if not content:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "content is required"})
                return
            if len(content) > MAX_CONTENT_LEN:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"content exceeds {MAX_CONTENT_LEN} characters"})
                return
            if len(sender_id) > MAX_ID_LEN or len(conversation_id) > MAX_ID_LEN:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"sender_id and conversation_id must be <= {MAX_ID_LEN} characters"})
                return

            message = IncomingMessage(sender_id=sender_id, conversation_id=conversation_id, content=content, timestamp=datetime.now(timezone.utc), platform="web")
            result = self.system.handle_message(message)
            response_data = result_to_dict(result)
            rl = self.system.rate_limiter
            if "rate_limited" in result.draft.safety_flags and rl is not None:
                response_data["retry_after_seconds"] = round(rl.retry_after(result.message.sender_id), 1)
            self._send_json(HTTPStatus.OK, response_data)
        except _BodyTooLargeError:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"request body exceeds {MAX_BODY_BYTES} bytes"})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})

    def _handle_local_ai_test(self) -> None:
        try:
            payload = self._read_json()
            base_url = str(payload.get("base_url", "http://127.0.0.1:8001")).strip()
            model = str(payload.get("model", "Qwen3.6-27B-4bit")).strip() or "Qwen3.6-27B-4bit"
            prompt = str(payload.get("prompt", "")).strip()
            api_key = str(payload.get("api_key", "")).strip()
            if not base_url or not prompt:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "base_url and prompt are required"})
                return
            if not any(base_url.startswith(s) for s in _ALLOWED_URL_SCHEMES):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "base_url must start with http:// or https://"})
                return
            result = call_local_ai(base_url, model, prompt, api_key)
            status = HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY
            self._send_json(status, result)
        except _BodyTooLargeError:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"request body exceeds {MAX_BODY_BYTES} bytes"})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0"))
        if length > MAX_BODY_BYTES:
            raise _BodyTooLargeError()
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        self._send_text(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

    def _send_text(self, status: HTTPStatus, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status.value)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("x-frame-options", "DENY")
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), WebAgentHandler)
    print(f"WebUI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
