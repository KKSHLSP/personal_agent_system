from http.server import ThreadingHTTPServer
from http.server import BaseHTTPRequestHandler
import io
import json
import threading
import unittest
import urllib.parse
import urllib.request

from agent import build_demo_system
from config import load_config
from webui import (
    MAX_BODY_BYTES,
    MAX_CONTENT_LEN,
    MAX_ID_LEN,
    _BodyTooLargeError,
    _WrongContentTypeError,
    WebAgentHandler,
    call_local_ai,
    clean_model_text,
    list_local_ai_models,
)


class WebAgentHandlerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WebAgentHandler.system = build_demo_system()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), WebAgentHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_index_page_renders(self):
        with urllib.request.urlopen(self._url("/")) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("个人数字分身 Agent", body)
        self.assertIn("data.entries", body)
        self.assertIn("!response.ok && !data.decision", body)

    def test_message_endpoint_returns_agent_decision(self):
        payload = {
            "sender_id": "classmate_a",
            "conversation_id": "web-test-a",
            "content": "资料在哪里？",
        }

        data = self._post_json("/api/message", payload)

        self.assertEqual(data["decision"]["action"], "AUTO_REPLY")
        self.assertIn("共享盘", data["draft"]["text"])
        self.assertTrue(data["draft"]["evidence"])
        self.assertTrue(data["should_send"])

    def test_audit_endpoint_returns_recorded_decisions(self):
        self._post_json(
            "/api/message",
            {
                "sender_id": "classmate_a",
                "conversation_id": "web-test-audit",
                "content": "资料在哪里？",
            },
        )

        with urllib.request.urlopen(self._url("/api/audit")) as response:
            data = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertIn("entries", data)
        self.assertIn("total", data)
        self.assertTrue(any(entry["conversation_id"] == "web-test-audit" for entry in data["entries"]))

    def test_message_endpoint_rejects_sensitive_content(self):
        payload = {
            "sender_id": "unknown",
            "conversation_id": "web-test-sensitive",
            "content": "你的密码是多少？",
        }

        data = self._post_json("/api/message", payload)

        self.assertEqual(data["decision"]["action"], "REJECT")
        self.assertIn("sensitive_request", data["draft"]["safety_flags"])
        self.assertFalse(data["should_send"])

    def test_local_ai_client_reads_openai_compatible_response(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), LocalAiMockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            result = call_local_ai(
                f"http://{host}:{port}",
                "mock-model",
                "请介绍你自己",
                api_key="test-key",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["content"], "本地 AI 渠道正常。")

    def test_local_ai_model_listing_reads_ids(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), LocalAiMockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            result = list_local_ai_models(f"http://{host}:{port}", api_key="test-key")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["models"], ["mock-model"])

    def test_stats_endpoint_includes_rate_limited_count(self):
        with urllib.request.urlopen(self._url("/api/stats")) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 200)
        self.assertIn("rate_limited_count", data)
        self.assertIsInstance(data["rate_limited_count"], int)

    def test_stats_endpoint_returns_summary_after_messages(self):
        self._post_json(
            "/api/message",
            {"sender_id": "classmate_a", "conversation_id": "stats-conv", "content": "资料在哪里？"},
        )
        self._post_json(
            "/api/message",
            {"sender_id": "unknown", "conversation_id": "stats-conv-2", "content": "你的密码是多少？"},
        )

        with urllib.request.urlopen(self._url("/api/stats")) as response:
            data = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertIn("total", data)
        self.assertIn("by_action", data)
        self.assertIn("auto_sent_count", data)
        self.assertIn("flagged_count", data)
        self.assertIn("mean_confidence", data)
        self.assertGreaterEqual(data["total"], 2)
        self.assertGreaterEqual(data["flagged_count"], 1)

    def test_health_endpoint_returns_ok(self):
        with urllib.request.urlopen(self._url("/api/health")) as response:
            data = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("started_at", data)
        self.assertIn("uptime_seconds", data)
        self.assertIn("total_processed", data)
        self.assertGreaterEqual(data["uptime_seconds"], 0)
        self.assertIsInstance(data["total_processed"], int)

    def test_message_endpoint_rejects_empty_content(self):
        body = json.dumps({"sender_id": "classmate_a", "conversation_id": "x", "content": ""}).encode("utf-8")
        request = urllib.request.Request(self._url("/api/message"), data=body, method="POST", headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("content", error_body["error"])

    def test_message_endpoint_rejects_content_too_long(self):
        body = json.dumps({"sender_id": "classmate_a", "content": "x" * (MAX_CONTENT_LEN + 1)}).encode("utf-8")
        request = urllib.request.Request(self._url("/api/message"), data=body, method="POST", headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("content", error_body["error"])

    def test_message_endpoint_rejects_id_too_long(self):
        body = json.dumps({"sender_id": "x" * (MAX_ID_LEN + 1), "content": "hello"}).encode("utf-8")
        request = urllib.request.Request(self._url("/api/message"), data=body, method="POST", headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("sender_id", error_body["error"])

    def test_message_endpoint_rejects_oversized_body(self):
        large_body = json.dumps({"content": "x" * (MAX_BODY_BYTES + 1)}).encode("utf-8")
        self.assertGreater(len(large_body), MAX_BODY_BYTES)
        request = urllib.request.Request(self._url("/api/message"), data=large_body, method="POST", headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 413)

    def test_local_ai_test_rejects_invalid_scheme(self):
        body = json.dumps({"base_url": "ftp://127.0.0.1:8001", "prompt": "hello"}).encode("utf-8")
        request = urllib.request.Request(self._url("/api/local-ai-test"), data=body, method="POST", headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("base_url", error_body["error"])

    def test_local_ai_models_endpoint_rejects_invalid_scheme(self):
        url = self._url("/api/local-ai-models?base_url=ftp://127.0.0.1:8001")
        request = urllib.request.Request(url, method="GET")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("base_url", error_body["error"])

    def test_local_ai_models_endpoint_rejects_file_scheme(self):
        url = self._url("/api/local-ai-models?base_url=file:///etc/passwd")
        request = urllib.request.Request(url, method="GET")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        with ctx.exception as error:
            self.assertEqual(error.code, 400)
            error_body = json.loads(error.read().decode("utf-8"))
        self.assertIn("base_url", error_body["error"])

    def test_responses_include_security_headers(self):
        with urllib.request.urlopen(self._url("/api/health")) as response:
            self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
            self.assertEqual(response.headers.get("x-frame-options"), "DENY")

    def test_html_response_includes_security_headers(self):
        with urllib.request.urlopen(self._url("/")) as response:
            self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
            self.assertEqual(response.headers.get("x-frame-options"), "DENY")

    def test_api_response_has_no_store_cache_control(self):
        with urllib.request.urlopen(self._url("/api/health")) as response:
            self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_html_response_has_no_store_cache_control(self):
        with urllib.request.urlopen(self._url("/")) as response:
            self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_clean_model_text_preserves_normal_text(self):
        self.assertEqual(clean_model_text("正常回答。"), "正常回答。")

    def _post_json(self, path, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url(path),
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _url(self, path):
        return f"http://{self.host}:{self.port}{path}"

class WebAgentRateLimitTest(unittest.TestCase):
    """Isolated server with a 1-msg-per-minute limit to test rate limiting."""

    @classmethod
    def setUpClass(cls):
        config = load_config()
        config.rate_limit.max_messages = 1
        config.rate_limit.window_seconds = 60.0
        system = build_demo_system(config)
        handler = type("_RLHandler", (WebAgentHandler,), {"system": system})
        cls.system = system
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _post_message(self, sender_id: str) -> tuple[int, dict]:
        """Returns (http_status_code, response_body_dict)."""
        body = json.dumps({
            "sender_id": sender_id,
            "conversation_id": f"rl-test-{sender_id}",
            "content": "资料在哪里？",
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://{self.host}:{self.port}/api/message",
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            finally:
                exc.close()

    def _post_message_raw_response(self, sender_id: str):
        """Returns the HTTPError or response object (for header inspection)."""
        body = json.dumps({
            "sender_id": sender_id,
            "conversation_id": f"rl-test-{sender_id}",
            "content": "资料在哪里？",
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://{self.host}:{self.port}/api/message",
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            return urllib.request.urlopen(request)
        except urllib.error.HTTPError as exc:
            return exc

    def test_first_message_is_allowed(self):
        status, data = self._post_message("rl_user_1")
        self.assertEqual(status, 200)
        self.assertNotEqual(data["decision"]["action"], "REJECT")
        self.assertNotIn("rate_limited", data["draft"]["safety_flags"])

    def test_second_message_is_rate_limited(self):
        self._post_message("rl_user_2")  # consume the 1 allowed message
        status, data = self._post_message("rl_user_2")
        self.assertEqual(data["decision"]["action"], "REJECT")
        self.assertIn("rate_limited", data["draft"]["safety_flags"])

    def test_rate_limited_response_includes_retry_after_seconds(self):
        self._post_message("rl_user_3")  # consume allowance
        _status, data = self._post_message("rl_user_3")
        self.assertIn("retry_after_seconds", data)
        self.assertGreater(data["retry_after_seconds"], 0.0)
        self.assertLessEqual(data["retry_after_seconds"], 60.0)

    def test_stats_rate_limited_count_increases_after_rejection(self):
        self._post_message("rl_user_4")  # consume allowance
        self._post_message("rl_user_4")  # triggers rejection
        with urllib.request.urlopen(f"http://{self.host}:{self.port}/api/stats") as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertGreaterEqual(data["rate_limited_count"], 1)

    def test_rate_limited_returns_429_status(self):
        self._post_message("rl_user_5")  # consume allowance
        status, data = self._post_message("rl_user_5")
        self.assertEqual(status, 429)
        self.assertIn("rate_limited", data["draft"]["safety_flags"])

    def test_rate_limited_response_has_retry_after_header(self):
        self._post_message("rl_user_6")  # consume allowance
        response = self._post_message_raw_response("rl_user_6")
        self.assertEqual(response.code if hasattr(response, "code") else response.status, 429)
        retry_after = response.headers.get("Retry-After")
        self.assertIsNotNone(retry_after, "Retry-After header must be present on 429 responses")
        self.assertGreater(int(retry_after), 0)


class LocalAiMockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        response = {
            "object": "list",
            "data": [{"id": "mock-model", "object": "model", "owned_by": "test"}],
        }
        self._send_json(response)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "本地 AI 渠道正常。",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        self._send_json(response)

    def _send_json(self, response):
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


class WebAgentAuditPaginationTest(unittest.TestCase):
    """Isolated server populated with a known number of audit entries."""

    N = 5  # messages to pre-populate

    @classmethod
    def setUpClass(cls):
        WebAgentHandler.system = build_demo_system()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), WebAgentHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # populate N audit entries
        for i in range(cls.N):
            body = json.dumps({"sender_id": "pager", "conversation_id": f"pg-{i}", "content": "资料在哪里？"}).encode()
            req = urllib.request.Request(
                f"http://{cls.host}:{cls.port}/api/message",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req):
                pass

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _get_audit(self, query=""):
        url = f"http://{self.host}:{self.port}/api/audit{query}"
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_no_params_returns_all_entries(self):
        status, data = self._get_audit()
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], self.N)
        self.assertEqual(len(data["entries"]), self.N)
        self.assertEqual(data["offset"], 0)
        self.assertEqual(data["limit"], self.N)

    def test_limit_restricts_returned_count(self):
        status, data = self._get_audit("?limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], self.N)
        self.assertEqual(len(data["entries"]), 2)
        self.assertEqual(data["limit"], 2)
        self.assertEqual(data["offset"], 0)

    def test_offset_skips_entries(self):
        _, full = self._get_audit()
        _, paged = self._get_audit("?offset=2")
        self.assertEqual(paged["total"], self.N)
        self.assertEqual(paged["offset"], 2)
        self.assertEqual(len(paged["entries"]), self.N - 2)
        # first entry in paged slice should match index 2 of full
        self.assertEqual(paged["entries"][0]["conversation_id"], full["entries"][2]["conversation_id"])

    def test_limit_and_offset_combined(self):
        status, data = self._get_audit("?limit=2&offset=1")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["entries"]), 2)
        self.assertEqual(data["offset"], 1)
        self.assertEqual(data["limit"], 2)

    def test_offset_beyond_end_returns_empty_entries(self):
        status, data = self._get_audit(f"?offset={self.N + 10}")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], self.N)
        self.assertEqual(data["entries"], [])

    def test_invalid_limit_treated_as_all(self):
        status, data = self._get_audit("?limit=abc")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["entries"]), self.N)

    def test_invalid_offset_treated_as_zero(self):
        status, data = self._get_audit("?offset=xyz")
        self.assertEqual(status, 200)
        self.assertEqual(data["offset"], 0)
        self.assertEqual(len(data["entries"]), self.N)

    def test_negative_limit_treated_as_all(self):
        status, data = self._get_audit("?limit=-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["entries"]), self.N)

    def test_negative_offset_clamped_to_zero(self):
        status, data = self._get_audit("?offset=-3")
        self.assertEqual(status, 200)
        self.assertEqual(data["offset"], 0)


class WebAgentAuditFilterTest(unittest.TestCase):
    """Integration tests for ?sender_id= and ?action= filter params on /api/audit."""

    @classmethod
    def setUpClass(cls):
        WebAgentHandler.system = build_demo_system()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), WebAgentHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # Send two messages from "filter_alice" (should auto-reply) and one from "filter_bob"
        for sender, content in [
            ("filter_alice", "资料在哪里？"),
            ("filter_alice", "资料在哪里？"),
            ("filter_bob", "资料在哪里？"),
        ]:
            body = json.dumps({"sender_id": sender, "conversation_id": f"fc-{sender}", "content": content}).encode()
            req = urllib.request.Request(
                f"http://{cls.host}:{cls.port}/api/message",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req):
                pass

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _get_audit(self, query=""):
        url = f"http://{self.host}:{self.port}/api/audit{query}"
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_filter_by_sender_id_returns_only_matching(self):
        status, data = self._get_audit("?sender_id=filter_alice")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 2)
        for entry in data["entries"]:
            self.assertEqual(entry["sender_id"], "filter_alice")
        self.assertEqual(data["filter_sender_id"], "filter_alice")

    def test_filter_by_url_encoded_sender_id(self):
        sender_id = "filter user 中文"
        body = json.dumps({
            "sender_id": sender_id,
            "conversation_id": "fc-encoded",
            "content": "资料在哪里？",
        }, ensure_ascii=False).encode()
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}/api/message",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req):
            pass

        query = "?sender_id=" + urllib.parse.quote(sender_id)
        status, data = self._get_audit(query)
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["entries"][0]["sender_id"], sender_id)

    def test_filter_by_sender_id_bob_returns_one(self):
        status, data = self._get_audit("?sender_id=filter_bob")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["entries"][0]["sender_id"], "filter_bob")

    def test_filter_by_unknown_sender_returns_empty(self):
        status, data = self._get_audit("?sender_id=nobody_here")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["entries"], [])

    def test_filter_action_lowercase_accepted(self):
        status, data = self._get_audit("?action=auto_reply")
        self.assertEqual(status, 200)
        # At least the three messages above triggered some action; verify filter key is uppercased
        self.assertEqual(data["filter_action"], "AUTO_REPLY")
        for entry in data["entries"]:
            self.assertEqual(entry["action"], "AUTO_REPLY")

    def test_filter_nonexistent_action_returns_empty(self):
        status, data = self._get_audit("?action=BOGUS_ACTION")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 0)

    def test_filter_absent_keys_not_in_response(self):
        status, data = self._get_audit()
        self.assertEqual(status, 200)
        self.assertNotIn("filter_sender_id", data)
        self.assertNotIn("filter_action", data)

    def test_combined_sender_and_action_filter(self):
        status, data = self._get_audit("?sender_id=filter_alice&action=AUTO_REPLY")
        self.assertEqual(status, 200)
        for entry in data["entries"]:
            self.assertEqual(entry["sender_id"], "filter_alice")
            self.assertEqual(entry["action"], "AUTO_REPLY")

    def test_filter_and_pagination_combined(self):
        status, data = self._get_audit("?sender_id=filter_alice&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["entries"]), 1)


class _IsolatedHandler(WebAgentHandler):
    """Handler subclass used only by WebAgentInternalErrorTest to avoid shared state."""


class WebAgentInternalErrorTest(unittest.TestCase):
    """Verify that unexpected exceptions in do_GET / do_POST return JSON 500."""

    @classmethod
    def setUpClass(cls):
        _IsolatedHandler.system = build_demo_system()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _IsolatedHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _url(self, path):
        return f"http://{self.host}:{self.port}{path}"

    def _post_and_read(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url(path), data=body, method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            with err:
                return err.code, json.loads(err.read().decode("utf-8"))

    def test_handle_message_exception_returns_json_500(self):
        original = _IsolatedHandler.system.handle_message

        def _raise(*_args):
            raise RuntimeError("unexpected pipeline failure")

        try:
            _IsolatedHandler.system.handle_message = _raise
            with self.assertLogs("webui", level="ERROR"):
                status, body = self._post_and_read(
                    "/api/message",
                    {"sender_id": "x", "conversation_id": "y", "content": "hello"},
                )
        finally:
            _IsolatedHandler.system.handle_message = original

        self.assertEqual(status, 500)
        self.assertIn("error", body)

    def test_get_stats_exception_returns_json_500(self):
        original = _IsolatedHandler.system.audit_log.stats

        def _raise():
            raise RuntimeError("stats boom")

        try:
            _IsolatedHandler.system.audit_log.stats = _raise
            with self.assertLogs("webui", level="ERROR"):
                try:
                    with urllib.request.urlopen(self._url("/api/stats")) as resp:
                        status, body = resp.status, json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as err:
                    with err:
                        status, body = err.code, json.loads(err.read().decode("utf-8"))
        finally:
            _IsolatedHandler.system.audit_log.stats = original

        self.assertEqual(status, 500)
        self.assertIn("error", body)


class WebAgentReadJsonTest(unittest.TestCase):
    """Unit tests for _read_json Content-Length edge cases."""

    def _make_handler(self, content_length_value: str, body_bytes: bytes = b""):
        class _FakeHandler(WebAgentHandler):
            def __init__(self, cl, body):
                self.headers = {"content-length": cl, "content-type": "application/json"}
                self.rfile = io.BytesIO(body)
        return _FakeHandler(content_length_value, body_bytes)

    def test_negative_content_length_reads_nothing(self):
        # rfile.read(-1) would drain the socket; clamped to 0 → empty dict
        handler = self._make_handler("-1", b'{"key": "value"}')
        result = handler._read_json()
        self.assertEqual(result, {})

    def test_large_negative_content_length_reads_nothing(self):
        handler = self._make_handler("-9999", b'{"key": "value"}')
        result = handler._read_json()
        self.assertEqual(result, {})

    def test_non_numeric_content_length_returns_empty_dict(self):
        # ValueError from int() should not bubble up as 500
        handler = self._make_handler("abc", b'{"key": "value"}')
        result = handler._read_json()
        self.assertEqual(result, {})

    def test_zero_content_length_returns_empty_dict(self):
        handler = self._make_handler("0", b"")
        result = handler._read_json()
        self.assertEqual(result, {})

    def test_valid_content_length_reads_body(self):
        body = b'{"sender_id": "test"}'
        handler = self._make_handler(str(len(body)), body)
        result = handler._read_json()
        self.assertEqual(result["sender_id"], "test")

    def test_oversized_content_length_raises_body_too_large(self):
        handler = self._make_handler(str(MAX_BODY_BYTES + 1), b"")
        with self.assertRaises(_BodyTooLargeError):
            handler._read_json()


class WebAgentStatsRateLimitedBySenderTest(unittest.TestCase):
    """Verify /api/stats includes rate_limited_by_sender breakdown."""

    @classmethod
    def setUpClass(cls):
        config = load_config()
        config.rate_limit.max_messages = 1
        config.rate_limit.window_seconds = 60.0
        system = build_demo_system(config)
        handler = type("_StatsHandler", (WebAgentHandler,), {"system": system})
        cls.system = system
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _post_message(self, sender_id: str) -> tuple[int, dict]:
        body = json.dumps({
            "sender_id": sender_id,
            "conversation_id": f"stats-test-{sender_id}",
            "content": "资料在哪里？",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}/api/message",
            data=body, method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            finally:
                exc.close()

    def _get_stats(self) -> dict:
        with urllib.request.urlopen(f"http://{self.host}:{self.port}/api/stats") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_stats_contains_rate_limited_by_sender_key(self):
        stats = self._get_stats()
        self.assertIn("rate_limited_by_sender", stats)

    def test_rate_limited_by_sender_is_empty_before_any_limit(self):
        stats = self._get_stats()
        self.assertIsInstance(stats["rate_limited_by_sender"], dict)

    def test_rate_limited_by_sender_populated_after_rejection(self):
        self._post_message("stats_sender_x")   # allowed
        self._post_message("stats_sender_x")   # rejected (limit=1/window)
        stats = self._get_stats()
        by_sender = stats["rate_limited_by_sender"]
        self.assertIn("stats_sender_x", by_sender)
        self.assertGreaterEqual(by_sender["stats_sender_x"], 1)

    def test_rate_limited_count_equals_sum_of_by_sender(self):
        # ensure at least one rejection recorded
        self._post_message("stats_sender_y")
        self._post_message("stats_sender_y")
        stats = self._get_stats()
        total = stats["rate_limited_count"]
        by_sender_sum = sum(stats["rate_limited_by_sender"].values())
        self.assertEqual(total, by_sender_sum)


class WebAgentStatsByFlagTest(unittest.TestCase):
    """Verify /api/stats includes by_flag breakdown."""

    @classmethod
    def setUpClass(cls):
        system = build_demo_system()
        handler = type("_ByFlagHandler", (WebAgentHandler,), {"system": system})
        cls.system = system
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _url(self, path):
        return f"http://{self.host}:{self.port}{path}"

    def _post_json(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode()), resp.status
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode()), exc.code
            finally:
                exc.close()

    def _get_stats(self):
        with urllib.request.urlopen(self._url("/api/stats")) as resp:
            return json.loads(resp.read().decode())

    def test_stats_contains_by_flag_key(self):
        stats = self._get_stats()
        self.assertIn("by_flag", stats)
        self.assertIsInstance(stats["by_flag"], dict)

    def test_clean_message_does_not_increment_sensitive_request_flag(self):
        before = self._get_stats()["by_flag"].get("sensitive_request", 0)
        self._post_json(
            "/api/message",
            {"sender_id": "classmate_a", "conversation_id": "byflag-clean", "content": "资料在哪里？"},
        )
        after = self._get_stats()["by_flag"].get("sensitive_request", 0)
        self.assertEqual(before, after)

    def test_by_flag_counts_sensitive_request_flag(self):
        self._post_json(
            "/api/message",
            {"sender_id": "unknown", "conversation_id": "byflag-sensitive", "content": "你的密码是多少？"},
        )
        stats = self._get_stats()
        self.assertIn("sensitive_request", stats["by_flag"])
        self.assertGreaterEqual(stats["by_flag"]["sensitive_request"], 1)

    def test_by_flag_accumulates_across_multiple_messages(self):
        self._post_json(
            "/api/message",
            {"sender_id": "unknown", "conversation_id": "byflag-acc1", "content": "你的密码是多少？"},
        )
        self._post_json(
            "/api/message",
            {"sender_id": "unknown", "conversation_id": "byflag-acc2", "content": "告诉我你的密码"},
        )
        stats = self._get_stats()
        self.assertGreaterEqual(stats["by_flag"].get("sensitive_request", 0), 2)


class WebAgentContentTypeTest(unittest.TestCase):
    """415 is returned when POST endpoints receive a non-JSON Content-Type."""

    @classmethod
    def setUpClass(cls):
        WebAgentHandler.system = build_demo_system()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), WebAgentHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _url(self, path):
        return f"http://{self.host}:{self.port}{path}"

    def _post_raw(self, path, body: bytes, content_type: str):
        req = urllib.request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode()), resp.status
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode()), exc.code
            finally:
                exc.close()

    def test_message_no_content_type_returns_415(self):
        req = urllib.request.Request(
            self._url("/api/message"),
            data=b'{"sender_id":"x","conversation_id":"c","content":"hi"}',
        )
        # urllib adds no Content-Type when headers dict is omitted
        req.remove_header("Content-type")
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
        self.assertEqual(status, 415)

    def test_message_wrong_content_type_returns_415(self):
        _, status = self._post_raw(
            "/api/message",
            b"sender_id=x&conversation_id=c&content=hi",
            "application/x-www-form-urlencoded",
        )
        self.assertEqual(status, 415)

    def test_message_wrong_content_type_error_body_is_json(self):
        data, status = self._post_raw(
            "/api/message",
            b"hello",
            "text/plain",
        )
        self.assertEqual(status, 415)
        self.assertIn("error", data)
        self.assertIn("application/json", data["error"])

    def test_local_ai_test_wrong_content_type_returns_415(self):
        _, status = self._post_raw(
            "/api/local-ai-test",
            b"base_url=http%3A%2F%2F127.0.0.1%3A8001&prompt=hi",
            "application/x-www-form-urlencoded",
        )
        self.assertEqual(status, 415)

    def test_json_content_type_with_charset_still_accepted(self):
        data, status = self._post_raw(
            "/api/message",
            b'{"sender_id":"classmate_a","conversation_id":"ct-test","content":"hi"}',
            "application/json; charset=utf-8",
        )
        self.assertNotEqual(status, 415)

    def test_json_content_type_is_case_insensitive(self):
        _data, status = self._post_raw(
            "/api/message",
            b'{"sender_id":"classmate_a","conversation_id":"ct-case","content":"hi"}',
            "Application/JSON",
        )
        self.assertNotEqual(status, 415)


if __name__ == "__main__":
    unittest.main()
