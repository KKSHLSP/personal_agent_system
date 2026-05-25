"""Tests for the config module and its wiring into agent components."""

from datetime import datetime, timezone
import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from config import (
    AgentConfig,
    AuditConfig,
    ConfidenceConfig,
    KnowledgeConfig,
    PermissionsConfig,
    RateLimitConfig,
    SafetyConfig,
    WebUIConfig,
    config_to_dict,
    load_config,
)
from agent import (
    IncomingMessage,
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeMatch,
    MessageRouter,
    PermissionPolicy,
    PermissionProfile,
    ReplyAction,
    SafetyPrivacyGuard,
    build_demo_system,
)


def _msg(content: str, sender_id: str = "alice") -> IncomingMessage:
    return IncomingMessage(sender_id, "conv-cfg", content, datetime.now(timezone.utc), "test")


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class ConfigDefaultsTest(unittest.TestCase):
    def setUp(self):
        self._config = load_config()

    def test_rate_limit_defaults(self):
        self.assertEqual(self._config.rate_limit.max_messages, 5)
        self.assertEqual(self._config.rate_limit.window_seconds, 60.0)

    def test_confidence_defaults(self):
        c = self._config.confidence
        self.assertAlmostEqual(c.auto_reply_threshold, 0.7)
        self.assertAlmostEqual(c.base_score, 0.55)
        self.assertAlmostEqual(c.score_multiplier, 0.1)
        self.assertAlmostEqual(c.cap, 0.95)

    def test_safety_defaults_include_chinese_and_english_patterns(self):
        patterns = self._config.safety.sensitive_patterns
        self.assertEqual(len(patterns), 2)
        self.assertTrue(any("密码" in p for p in patterns))
        self.assertTrue(any("password" in p for p in patterns))

    def test_knowledge_defaults(self):
        self.assertEqual(self._config.knowledge.search_limit, 3)
        self.assertEqual(self._config.knowledge.max_recent_messages, 10)
        self.assertEqual(self._config.knowledge.knowledge_file, "")

    def test_webui_defaults(self):
        w = self._config.webui
        self.assertEqual(w.host, "127.0.0.1")
        self.assertEqual(w.port, 8000)
        self.assertIn("8001", w.local_ai_url)

    def test_no_path_returns_defaults(self):
        config = load_config(None)
        self.assertEqual(config.rate_limit.max_messages, 5)

    def test_config_sections_are_independent_instances(self):
        c1 = load_config()
        c2 = load_config()
        c1.rate_limit.max_messages = 999
        self.assertEqual(c2.rate_limit.max_messages, 5)


# ---------------------------------------------------------------------------
# JSON file loading
# ---------------------------------------------------------------------------

class ConfigJsonLoadingTest(unittest.TestCase):
    def _write_json(self, data: dict) -> str:
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, fp, ensure_ascii=False)
        fp.close()
        return fp.name

    def tearDown(self):
        pass  # individual tests clean up their own files

    def test_load_rate_limit_from_json(self):
        path = self._write_json({"rate_limit": {"max_messages": 20, "window_seconds": 30.0}})
        try:
            config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 20)
            self.assertAlmostEqual(config.rate_limit.window_seconds, 30.0)
        finally:
            os.unlink(path)

    def test_partial_json_leaves_unspecified_fields_at_default(self):
        path = self._write_json({"rate_limit": {"max_messages": 15}})
        try:
            config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 15)
            self.assertAlmostEqual(config.rate_limit.window_seconds, 60.0)  # untouched
        finally:
            os.unlink(path)

    def test_partial_json_leaves_unspecified_sections_at_default(self):
        path = self._write_json({"webui": {"port": 9090}})
        try:
            config = load_config(path)
            self.assertEqual(config.webui.port, 9090)
            self.assertEqual(config.rate_limit.max_messages, 5)  # untouched
        finally:
            os.unlink(path)

    def test_missing_file_returns_defaults(self):
        config = load_config("/nonexistent/path/agent.json")
        self.assertEqual(config.rate_limit.max_messages, 5)

    def test_unknown_section_in_json_is_ignored(self):
        path = self._write_json({"totally_unknown": {"foo": "bar"}})
        try:
            config = load_config(path)  # must not raise
            self.assertEqual(config.rate_limit.max_messages, 5)
        finally:
            os.unlink(path)

    def test_unknown_key_within_section_is_ignored(self):
        path = self._write_json({"rate_limit": {"max_messages": 10, "no_such_key": 99}})
        try:
            config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 10)
        finally:
            os.unlink(path)

    def test_knowledge_file_loadable_via_json(self):
        path = self._write_json({"knowledge": {"knowledge_file": "/data/my_kb.json"}})
        try:
            config = load_config(path)
            self.assertEqual(config.knowledge.knowledge_file, "/data/my_kb.json")
        finally:
            os.unlink(path)

    def test_safety_patterns_overridable_via_json(self):
        path = self._write_json({"safety": {"sensitive_patterns": [r"my_secret_word"]}})
        try:
            config = load_config(path)
            self.assertEqual(config.safety.sensitive_patterns, [r"my_secret_word"])
        finally:
            os.unlink(path)

    def test_all_confidence_fields_loadable(self):
        path = self._write_json({
            "confidence": {
                "auto_reply_threshold": 0.85,
                "base_score": 0.6,
                "score_multiplier": 0.15,
                "cap": 0.98,
            }
        })
        try:
            config = load_config(path)
            self.assertAlmostEqual(config.confidence.auto_reply_threshold, 0.85)
            self.assertAlmostEqual(config.confidence.base_score, 0.6)
            self.assertAlmostEqual(config.confidence.score_multiplier, 0.15)
            self.assertAlmostEqual(config.confidence.cap, 0.98)
        finally:
            os.unlink(path)

    def test_empty_json_file_returns_defaults(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        fp.write("{}")
        fp.close()
        try:
            config = load_config(fp.name)
            self.assertEqual(config.rate_limit.max_messages, 5)
        finally:
            os.unlink(fp.name)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

_ENV_KEYS = [
    "AGENT_RATE_LIMIT_MAX_MESSAGES",
    "AGENT_RATE_LIMIT_WINDOW_SECONDS",
    "AGENT_CONFIDENCE_AUTO_REPLY_THRESHOLD",
    "AGENT_CONFIDENCE_BASE_SCORE",
    "AGENT_CONFIDENCE_SCORE_MULTIPLIER",
    "AGENT_CONFIDENCE_CAP",
    "AGENT_KNOWLEDGE_SEARCH_LIMIT",
    "AGENT_KNOWLEDGE_MAX_RECENT_MESSAGES",
    "AGENT_KNOWLEDGE_FILE",
    "AGENT_WEBUI_HOST",
    "AGENT_WEBUI_PORT",
    "AGENT_WEBUI_LOCAL_AI_URL",
]


class ConfigEnvVarTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k) for k in _ENV_KEYS if k in os.environ}

    def tearDown(self):
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(self._saved)

    def test_env_overrides_rate_limit_max_messages(self):
        os.environ["AGENT_RATE_LIMIT_MAX_MESSAGES"] = "42"
        self.assertEqual(load_config().rate_limit.max_messages, 42)

    def test_env_overrides_rate_limit_window_seconds(self):
        os.environ["AGENT_RATE_LIMIT_WINDOW_SECONDS"] = "120.5"
        self.assertAlmostEqual(load_config().rate_limit.window_seconds, 120.5)

    def test_env_overrides_confidence_threshold(self):
        os.environ["AGENT_CONFIDENCE_AUTO_REPLY_THRESHOLD"] = "0.9"
        self.assertAlmostEqual(load_config().confidence.auto_reply_threshold, 0.9)

    def test_env_overrides_webui_port(self):
        os.environ["AGENT_WEBUI_PORT"] = "9999"
        self.assertEqual(load_config().webui.port, 9999)

    def test_env_overrides_webui_host(self):
        os.environ["AGENT_WEBUI_HOST"] = "0.0.0.0"
        self.assertEqual(load_config().webui.host, "0.0.0.0")

    def test_env_overrides_knowledge_search_limit(self):
        os.environ["AGENT_KNOWLEDGE_SEARCH_LIMIT"] = "7"
        self.assertEqual(load_config().knowledge.search_limit, 7)

    def test_env_overrides_knowledge_max_recent_messages(self):
        os.environ["AGENT_KNOWLEDGE_MAX_RECENT_MESSAGES"] = "25"
        self.assertEqual(load_config().knowledge.max_recent_messages, 25)

    def test_env_overrides_knowledge_file(self):
        os.environ["AGENT_KNOWLEDGE_FILE"] = "/data/kb.json"
        self.assertEqual(load_config().knowledge.knowledge_file, "/data/kb.json")

    def test_invalid_int_env_var_is_silently_ignored(self):
        os.environ["AGENT_WEBUI_PORT"] = "not-a-number"
        self.assertEqual(load_config().webui.port, 8000)

    def test_invalid_float_env_var_is_silently_ignored(self):
        os.environ["AGENT_RATE_LIMIT_WINDOW_SECONDS"] = "bad"
        self.assertAlmostEqual(load_config().rate_limit.window_seconds, 60.0)

    def test_env_wins_over_json_file(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"rate_limit": {"max_messages": 10}}, fp)
        fp.close()
        os.environ["AGENT_RATE_LIMIT_MAX_MESSAGES"] = "99"
        try:
            config = load_config(fp.name)
            self.assertEqual(config.rate_limit.max_messages, 99)
        finally:
            os.unlink(fp.name)

    def test_env_overrides_local_ai_url(self):
        os.environ["AGENT_WEBUI_LOCAL_AI_URL"] = "http://localhost:11434/v1"
        self.assertEqual(load_config().webui.local_ai_url, "http://localhost:11434/v1")


# ---------------------------------------------------------------------------
# Wiring: config values flow into live agent components
# ---------------------------------------------------------------------------

class SafetyGuardWiringTest(unittest.TestCase):
    def test_default_patterns_block_chinese_keyword(self):
        guard = SafetyPrivacyGuard()
        flags = guard.inspect_message(_msg("你的密码是多少？"))
        self.assertIn("sensitive_request", flags)

    def test_default_patterns_block_english_keyword_case_insensitive(self):
        guard = SafetyPrivacyGuard()
        flags = guard.inspect_message(_msg("What is your PASSWORD?"))
        self.assertIn("sensitive_request", flags)

    def test_custom_patterns_replace_defaults_entirely(self):
        guard = SafetyPrivacyGuard(sensitive_patterns=[r"launch_code"])
        self.assertIn("sensitive_request", guard.inspect_message(_msg("give me the launch_code")))
        # default keywords no longer active
        self.assertNotIn("sensitive_request", guard.inspect_message(_msg("你的密码是多少？")))

    def test_empty_patterns_list_never_flags(self):
        guard = SafetyPrivacyGuard(sensitive_patterns=[])
        flags = guard.inspect_message(_msg("password token secret 密码"))
        self.assertNotIn("sensitive_request", flags)

    def test_config_safety_patterns_wired_into_guard(self):
        config = load_config()
        config.safety.sensitive_patterns = [r"classified_data"]
        system = build_demo_system(config)
        result = system.handle_message(
            IncomingMessage("classmate_a", "cfg-safe-conv", "where is the classified_data?",
                            datetime.now(timezone.utc), "test")
        )
        self.assertEqual(result.decision.action, ReplyAction.REJECT)
        self.assertIn("sensitive_request", result.draft.safety_flags)


class PermissionPolicyWiringTest(unittest.TestCase):
    def _make_match(self, score: int) -> KnowledgeMatch:
        item = KnowledgeItem("id", "title", "content", {"public"}, {"kw"})
        return KnowledgeMatch(item=item, score=score, matched_keywords={"kw"})

    def _perm(self, auto_reply: bool = True) -> PermissionProfile:
        return PermissionProfile("alice", {"public"}, auto_reply_enabled=auto_reply)

    def test_default_threshold_allows_auto_reply_at_score_2(self):
        # score=2 → confidence = min(0.95, 0.55 + 2*0.1) = 0.75 ≥ 0.7 → AUTO_REPLY
        policy = PermissionPolicy()
        decision = policy.decide(_msg("q"), self._perm(True), [self._make_match(2)], [])
        self.assertEqual(decision.action, ReplyAction.AUTO_REPLY)

    def test_high_threshold_blocks_auto_reply_at_same_score(self):
        # same score=2 → confidence=0.75 < 0.9 → SUGGEST_ONLY
        policy = PermissionPolicy(auto_reply_threshold=0.9)
        decision = policy.decide(_msg("q"), self._perm(True), [self._make_match(2)], [])
        self.assertEqual(decision.action, ReplyAction.SUGGEST_ONLY)

    def test_low_threshold_auto_replies_at_score_1(self):
        # score=1 → confidence = min(0.95, 0.55 + 1*0.1) = 0.65 ≥ 0.6 → AUTO_REPLY
        policy = PermissionPolicy(auto_reply_threshold=0.6)
        decision = policy.decide(_msg("q"), self._perm(True), [self._make_match(1)], [])
        self.assertEqual(decision.action, ReplyAction.AUTO_REPLY)

    def test_custom_cap_limits_confidence(self):
        policy = PermissionPolicy(cap=0.7, auto_reply_threshold=0.65)
        # score=40 → uncapped = 0.55 + 4.0 > 0.7, capped to 0.7
        decision = policy.decide(_msg("q"), self._perm(True), [self._make_match(40)], [])
        self.assertAlmostEqual(decision.confidence, 0.7)

    def test_config_confidence_threshold_wired_into_policy(self):
        config = load_config()
        config.confidence.auto_reply_threshold = 0.999  # impossible to reach
        system = build_demo_system(config)
        result = system.handle_message(
            IncomingMessage("classmate_a", "cfg-conf-conv", "资料在哪里？",
                            datetime.now(timezone.utc), "test")
        )
        # classmate_a has auto_reply_enabled=True, but threshold unreachable → SUGGEST_ONLY
        self.assertEqual(result.decision.action, ReplyAction.SUGGEST_ONLY)


class MessageRouterWiringTest(unittest.TestCase):
    def _perm(self) -> PermissionProfile:
        return PermissionProfile("user", {"public"}, auto_reply_enabled=False)

    def test_custom_max_recent_messages_bounds_history(self):
        router = MessageRouter(max_recent_messages=3)
        perm = self._perm()
        for i in range(7):
            msg = IncomingMessage("user", "conv", f"msg-{i}", datetime.now(timezone.utc), "mock")
            ctx = router.get_context(msg, perm)
        self.assertLessEqual(len(ctx.recent_messages), 3)

    def test_default_max_recent_messages_is_10(self):
        router = MessageRouter()
        perm = self._perm()
        for i in range(15):
            msg = IncomingMessage("user", "conv", f"msg-{i}", datetime.now(timezone.utc), "mock")
            ctx = router.get_context(msg, perm)
        self.assertLessEqual(len(ctx.recent_messages), 10)

    def test_config_max_recent_messages_wired_into_build_demo(self):
        config = load_config()
        config.knowledge.max_recent_messages = 2
        system = build_demo_system(config)
        for i in range(5):
            system.handle_message(
                IncomingMessage("classmate_a", "cfg-router-conv", f"资料在哪里 {i}？",
                                datetime.now(timezone.utc), "test")
            )
        ctx = system.router._contexts.get("cfg-router-conv")
        self.assertIsNotNone(ctx)
        self.assertLessEqual(len(ctx.recent_messages), 2)


class KnowledgeSearchLimitTest(unittest.TestCase):
    def _make_kb(self) -> KnowledgeBase:
        return KnowledgeBase([
            KnowledgeItem(f"item_{i}", f"课程资料 {i}", f"内容 {i}", {"public"}, {"资料", f"词{i}"})
            for i in range(10)
        ])

    def test_search_limit_1_returns_at_most_one_result(self):
        kb = self._make_kb()
        matches = kb.search("资料", {"public"}, limit=1)
        self.assertLessEqual(len(matches), 1)

    def test_search_limit_5_returns_at_most_five(self):
        kb = self._make_kb()
        matches = kb.search("资料", {"public"}, limit=5)
        self.assertLessEqual(len(matches), 5)

    def test_search_default_limit_is_3(self):
        kb = self._make_kb()
        matches = kb.search("资料", {"public"})
        self.assertLessEqual(len(matches), 3)

    def test_config_search_limit_wired_into_build_demo(self):
        config = load_config()
        config.knowledge.search_limit = 1
        system = build_demo_system(config)

        result = system.handle_message(
            IncomingMessage("classmate_b", "cfg-search-limit", "资料 作业",
                            datetime.now(timezone.utc), "test")
        )

        self.assertLessEqual(len(result.draft.evidence), 2)


class AuditConfigTest(unittest.TestCase):
    _ENV_KEYS = ["AGENT_AUDIT_LOG_FILE", "AGENT_AUDIT_MAX_ENTRIES"]

    def setUp(self):
        self._saved = {k: os.environ.pop(k) for k in self._ENV_KEYS if k in os.environ}

    def tearDown(self):
        for k in self._ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(self._saved)

    def test_audit_default_log_file_is_empty(self):
        config = load_config()
        self.assertEqual(config.audit.log_file, "")

    def test_audit_log_file_from_json(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"audit": {"log_file": "/tmp/test_audit.jsonl"}}, fp)
        fp.close()
        try:
            config = load_config(fp.name)
            self.assertEqual(config.audit.log_file, "/tmp/test_audit.jsonl")
        finally:
            os.unlink(fp.name)

    def test_env_overrides_audit_log_file(self):
        os.environ["AGENT_AUDIT_LOG_FILE"] = "/var/log/agent.jsonl"
        config = load_config()
        self.assertEqual(config.audit.log_file, "/var/log/agent.jsonl")

    def test_env_audit_wins_over_json(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"audit": {"log_file": "/tmp/from_json.jsonl"}}, fp)
        fp.close()
        os.environ["AGENT_AUDIT_LOG_FILE"] = "/tmp/from_env.jsonl"
        try:
            config = load_config(fp.name)
            self.assertEqual(config.audit.log_file, "/tmp/from_env.jsonl")
        finally:
            os.unlink(fp.name)

    def test_audit_config_is_independent_across_load_calls(self):
        c1 = load_config()
        c2 = load_config()
        c1.audit.log_file = "/tmp/mutated.jsonl"
        self.assertEqual(c2.audit.log_file, "")

    def test_config_to_dict_redacts_local_ai_url_userinfo(self):
        config = AgentConfig()
        config.webui.local_ai_url = "http://user:secret@127.0.0.1:11434/v1?x=1"
        data = config_to_dict(config)
        self.assertEqual(data["webui"]["local_ai_url"], "http://***@127.0.0.1:11434/v1?x=1")
        self.assertNotIn("secret", data["webui"]["local_ai_url"])

    def test_config_to_dict_preserves_url_without_userinfo(self):
        config = AgentConfig()
        config.webui.local_ai_url = "http://127.0.0.1:11434/v1"
        data = config_to_dict(config)
        self.assertEqual(data["webui"]["local_ai_url"], "http://127.0.0.1:11434/v1")

    def test_config_to_dict_includes_sections(self):
        data = config_to_dict(AgentConfig())
        for section in ("rate_limit", "confidence", "safety", "knowledge", "audit", "permissions", "webui"):
            self.assertIn(section, data)

    def test_audit_default_max_entries_is_zero(self):
        config = load_config()
        self.assertEqual(config.audit.max_entries, 0)

    def test_audit_max_entries_from_json(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"audit": {"max_entries": 500}}, fp)
        fp.close()
        try:
            config = load_config(fp.name)
            self.assertEqual(config.audit.max_entries, 500)
        finally:
            os.unlink(fp.name)

    def test_env_overrides_max_entries(self):
        os.environ["AGENT_AUDIT_MAX_ENTRIES"] = "200"
        config = load_config()
        self.assertEqual(config.audit.max_entries, 200)

    def test_env_max_entries_wins_over_json(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"audit": {"max_entries": 100}}, fp)
        fp.close()
        os.environ["AGENT_AUDIT_MAX_ENTRIES"] = "999"
        try:
            config = load_config(fp.name)
            self.assertEqual(config.audit.max_entries, 999)
        finally:
            os.unlink(fp.name)

    def test_build_demo_system_passes_log_file_to_audit_log(self):
        import tempfile as _tmp
        fp = _tmp.NamedTemporaryFile(suffix=".jsonl", delete=False)
        fp.close()
        os.unlink(fp.name)
        try:
            config = load_config()
            config.audit.log_file = fp.name
            from agent import IncomingMessage, build_demo_system
            system = build_demo_system(config)
            system.handle_message(
                IncomingMessage("classmate_a", "audit-cfg-conv", "资料在哪里？",
                                datetime.now(timezone.utc), "test")
            )
            self.assertTrue(os.path.exists(fp.name))
            with open(fp.name, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            self.assertEqual(len(lines), 1)
            import json as _json
            self.assertEqual(_json.loads(lines[0])["action"], "AUTO_REPLY")
        finally:
            try:
                os.unlink(fp.name)
            except FileNotFoundError:
                pass


class RateLimitConfigWiringTest(unittest.TestCase):
    def test_config_rate_limit_wired_into_build_demo(self):
        config = load_config()
        config.rate_limit.max_messages = 1
        config.rate_limit.window_seconds = 60.0
        system = build_demo_system(config)

        system.handle_message(
            IncomingMessage("classmate_a", "cfg-rate-limit-1", "资料在哪里？",
                            datetime.now(timezone.utc), "test")
        )
        result = system.handle_message(
            IncomingMessage("classmate_a", "cfg-rate-limit-2", "作业什么时候提交？",
                            datetime.now(timezone.utc), "test")
        )

        self.assertEqual(result.decision.action, ReplyAction.REJECT)
        self.assertIn("rate_limited", result.draft.safety_flags)


class PermissionsConfigTest(unittest.TestCase):
    def test_default_permissions_file_is_empty(self):
        config = AgentConfig()
        self.assertEqual(config.permissions.permissions_file, "")

    def test_permissions_config_is_present_in_agent_config(self):
        config = AgentConfig()
        self.assertIsInstance(config.permissions, PermissionsConfig)

    def test_permissions_file_loaded_from_json_file(self):
        fp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        )
        json.dump({"rate_limit": {"max_messages": 5}, "permissions": {"permissions_file": "/tmp/p.json"}}, fp)
        fp.close()
        try:
            config = load_config(fp.name)
            self.assertEqual(config.permissions.permissions_file, "/tmp/p.json")
        finally:
            os.unlink(fp.name)

    def test_permissions_file_env_var_overrides(self):
        os.environ["AGENT_PERMISSIONS_FILE"] = "/env/permissions.json"
        try:
            config = load_config()
            self.assertEqual(config.permissions.permissions_file, "/env/permissions.json")
        finally:
            del os.environ["AGENT_PERMISSIONS_FILE"]

    def test_env_var_permissions_file_empty_string_is_valid(self):
        os.environ["AGENT_PERMISSIONS_FILE"] = ""
        try:
            config = load_config()
            self.assertEqual(config.permissions.permissions_file, "")
        finally:
            del os.environ["AGENT_PERMISSIONS_FILE"]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class ConfigValidationTest(unittest.TestCase):
    """validate() enforces value ranges; load_config() raises on invalid files."""

    def test_default_config_passes_validation(self):
        AgentConfig().validate()

    def test_load_config_defaults_pass_validation(self):
        load_config()

    def test_max_messages_zero_raises(self):
        c = AgentConfig()
        c.rate_limit.max_messages = 0
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("max_messages", str(ctx.exception))

    def test_max_messages_negative_raises(self):
        c = AgentConfig()
        c.rate_limit.max_messages = -1
        with self.assertRaises(ValueError):
            c.validate()

    def test_window_seconds_zero_raises(self):
        c = AgentConfig()
        c.rate_limit.window_seconds = 0.0
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("window_seconds", str(ctx.exception))

    def test_window_seconds_negative_raises(self):
        c = AgentConfig()
        c.rate_limit.window_seconds = -1.0
        with self.assertRaises(ValueError):
            c.validate()

    def test_auto_reply_threshold_above_one_raises(self):
        c = AgentConfig()
        c.confidence.auto_reply_threshold = 1.1
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("auto_reply_threshold", str(ctx.exception))

    def test_auto_reply_threshold_negative_raises(self):
        c = AgentConfig()
        c.confidence.auto_reply_threshold = -0.01
        with self.assertRaises(ValueError):
            c.validate()

    def test_base_score_above_one_raises(self):
        c = AgentConfig()
        c.confidence.base_score = 1.5
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("base_score", str(ctx.exception))

    def test_cap_negative_raises(self):
        c = AgentConfig()
        c.confidence.cap = -0.1
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("cap", str(ctx.exception))

    def test_score_multiplier_negative_raises(self):
        c = AgentConfig()
        c.confidence.score_multiplier = -0.5
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("score_multiplier", str(ctx.exception))

    def test_score_multiplier_zero_is_valid(self):
        c = AgentConfig()
        c.confidence.score_multiplier = 0.0
        c.validate()

    def test_search_limit_zero_raises(self):
        c = AgentConfig()
        c.knowledge.search_limit = 0
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("search_limit", str(ctx.exception))

    def test_max_recent_messages_zero_raises(self):
        c = AgentConfig()
        c.knowledge.max_recent_messages = 0
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("max_recent_messages", str(ctx.exception))

    def test_port_zero_raises(self):
        c = AgentConfig()
        c.webui.port = 0
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("port", str(ctx.exception))

    def test_port_65536_raises(self):
        c = AgentConfig()
        c.webui.port = 65536
        with self.assertRaises(ValueError):
            c.validate()

    def test_load_config_with_invalid_json_raises(self):
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"rate_limit": {"max_messages": 0}}, fp)
        fp.close()
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(fp.name)
            self.assertIn("max_messages", str(ctx.exception))
        finally:
            os.unlink(fp.name)

    def test_boundary_values_are_valid(self):
        c = AgentConfig()
        c.rate_limit.max_messages = 1
        c.rate_limit.window_seconds = 0.001
        c.confidence.auto_reply_threshold = 0.0
        c.confidence.base_score = 1.0
        c.confidence.cap = 0.0
        c.confidence.score_multiplier = 0.0
        c.knowledge.search_limit = 1
        c.knowledge.max_recent_messages = 1
        c.webui.port = 1
        c.validate()

    def test_port_65535_is_valid(self):
        c = AgentConfig()
        c.webui.port = 65535
        c.validate()

    def test_audit_max_entries_negative_raises(self):
        c = AgentConfig()
        c.audit.max_entries = -1
        with self.assertRaises(ValueError) as ctx:
            c.validate()
        self.assertIn("max_entries", str(ctx.exception))

    def test_audit_max_entries_zero_is_valid(self):
        c = AgentConfig()
        c.audit.max_entries = 0
        c.validate()

    def test_audit_max_entries_positive_is_valid(self):
        c = AgentConfig()
        c.audit.max_entries = 10000
        c.validate()


class ConfigToDictRedactionTest(unittest.TestCase):
    """config_to_dict must not leak internal paths or detection-rule content."""

    def test_safety_pattern_count_returned_not_patterns(self):
        data = config_to_dict(AgentConfig())
        self.assertNotIn("sensitive_patterns", data["safety"])
        self.assertIn("pattern_count", data["safety"])
        self.assertIsInstance(data["safety"]["pattern_count"], int)
        self.assertGreater(data["safety"]["pattern_count"], 0)

    def test_safety_pattern_count_reflects_actual_count(self):
        config = AgentConfig()
        config.safety.sensitive_patterns = ["a", "b", "c"]
        data = config_to_dict(config)
        self.assertEqual(data["safety"]["pattern_count"], 3)

    def test_audit_file_path_not_exposed(self):
        config = AgentConfig()
        config.audit.log_file = "/var/log/agent.jsonl"
        data = config_to_dict(config)
        self.assertNotIn("log_file", data["audit"])
        self.assertNotIn("/var/log", str(data))

    def test_audit_file_enabled_true_when_path_set(self):
        config = AgentConfig()
        config.audit.log_file = "/var/log/agent.jsonl"
        self.assertTrue(config_to_dict(config)["audit"]["file_enabled"])

    def test_audit_file_enabled_false_when_empty(self):
        self.assertFalse(config_to_dict(AgentConfig())["audit"]["file_enabled"])

    def test_knowledge_file_path_not_exposed(self):
        config = AgentConfig()
        config.knowledge.knowledge_file = "/data/kb.json"
        data = config_to_dict(config)
        self.assertNotIn("knowledge_file", data["knowledge"])
        self.assertNotIn("/data", str(data))

    def test_knowledge_file_enabled_true_when_path_set(self):
        config = AgentConfig()
        config.knowledge.knowledge_file = "/data/kb.json"
        self.assertTrue(config_to_dict(config)["knowledge"]["file_enabled"])

    def test_knowledge_file_enabled_false_when_empty(self):
        self.assertFalse(config_to_dict(AgentConfig())["knowledge"]["file_enabled"])

    def test_permissions_file_path_not_exposed(self):
        config = AgentConfig()
        config.permissions.permissions_file = "/etc/agent/perms.json"
        data = config_to_dict(config)
        self.assertNotIn("permissions_file", data["permissions"])
        self.assertNotIn("/etc", str(data))

    def test_permissions_file_enabled_true_when_path_set(self):
        config = AgentConfig()
        config.permissions.permissions_file = "/etc/agent/perms.json"
        self.assertTrue(config_to_dict(config)["permissions"]["file_enabled"])

    def test_permissions_file_enabled_false_when_empty(self):
        self.assertFalse(config_to_dict(AgentConfig())["permissions"]["file_enabled"])


class LoadFileOSErrorTest(unittest.TestCase):
    """_load_file must silently fall back to defaults on OSError after exists()."""

    def _write_json_file(self, data: dict) -> str:
        fp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, fp)
        fp.close()
        return fp.name

    def test_permission_denied_returns_defaults(self):
        path = self._write_json_file({"rate_limit": {"max_messages": 99}})
        try:
            with patch("builtins.open", side_effect=PermissionError("denied")):
                config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 5)
        finally:
            os.unlink(path)

    def test_oserror_returns_defaults(self):
        path = self._write_json_file({"rate_limit": {"max_messages": 99}})
        try:
            with patch("builtins.open", side_effect=OSError("io error")):
                config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 5)
        finally:
            os.unlink(path)

    def test_oserror_logs_warning(self):
        path = self._write_json_file({})
        try:
            with patch("builtins.open", side_effect=PermissionError("denied")):
                with self.assertLogs("config", level=logging.WARNING) as cm:
                    load_config(path)
            self.assertTrue(
                any("Cannot read config file" in line for line in cm.output),
                cm.output,
            )
        finally:
            os.unlink(path)

    def test_readable_file_still_loads_normally(self):
        path = self._write_json_file({"rate_limit": {"max_messages": 7}})
        try:
            config = load_config(path)
            self.assertEqual(config.rate_limit.max_messages, 7)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
