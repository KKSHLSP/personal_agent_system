from http.server import ThreadingHTTPServer
from http.server import BaseHTTPRequestHandler
import json
import threading
import unittest
import urllib.request

from agent import build_demo_system
from webui import WebAgentHandler, call_local_ai, clean_model_text, list_local_ai_models


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
        self.assertTrue(any(entry["conversation_id"] == "web-test-audit" for entry in data))

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


if __name__ == "__main__":
    unittest.main()
