from datetime import datetime, timezone
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from agent import (
    AuditEntry,
    AuditLog,
    AuditStats,
    ConversationContext,
    IncomingMessage,
    KnowledgeBase,
    KnowledgeItem,
    MessageRouter,
    MockChatAdapter,
    PermissionPolicy,
    PermissionProfile,
    PermissionStore,
    PersonalAgentSystem,
    RateLimiter,
    ReplyAction,
    ReplyGenerator,
    SafetyPrivacyGuard,
    StyleProfile,
    build_demo_system,
    _parse_iso_dt,
    _tokenize,
)


def make_system(auto_reply_enabled=True, allowed_tags=None, contact_id="alice", rate_limiter=None):
    allowed_tags = allowed_tags or {"public", "course"}
    adapter = MockChatAdapter()
    permissions = PermissionStore(
        {
            contact_id: PermissionProfile(
                contact_id=contact_id,
                allowed_knowledge_tags=allowed_tags,
                auto_reply_enabled=auto_reply_enabled,
            )
        }
    )
    knowledge = KnowledgeBase(
        [
            KnowledgeItem(
                "material",
                "课程资料位置",
                "课程资料在班级共享盘的 Week 8 文件夹里，请查看最新版本。",
                {"public", "course"},
                {"资料", "哪里", "共享盘"},
            ),
            KnowledgeItem(
                "deadline",
                "作业提交时间",
                "作业截止时间是本周五 23:59，提交到课程平台。",
                {"public", "course"},
                {"作业", "提交", "什么时候", "截止"},
            ),
            KnowledgeItem(
                "private",
                "私人聊天摘要",
                "这份内容只允许本人确认后发送。",
                {"private"},
                {"私人", "聊天记录"},
                sensitivity="restricted",
            ),
        ]
    )
    style = StyleProfile({contact_id: ["口语", "简短"]})
    return PersonalAgentSystem(
        adapter,
        permissions,
        MessageRouter(),
        knowledge,
        PermissionPolicy(),
        ReplyGenerator(style),
        SafetyPrivacyGuard(),
        AuditLog(),
        rate_limiter=rate_limiter,
    )


def message(content, sender_id="alice", conversation_id="conv-a"):
    return IncomingMessage(
        sender_id=sender_id,
        conversation_id=conversation_id,
        content=content,
        timestamp=datetime.now(timezone.utc),
        platform="mock",
    )


class PersonalAgentSystemTest(unittest.TestCase):
    def test_repeated_question_uses_knowledge_and_evidence(self):
        system = make_system(auto_reply_enabled=True)

        result = system.handle_message(message("资料在哪里？"))

        self.assertEqual(result.decision.action, ReplyAction.AUTO_REPLY)
        self.assertIn("共享盘", result.draft.text)
        self.assertTrue(result.draft.evidence)
        self.assertEqual(system.adapter.sent_replies, [("conv-a", result.draft.text)])

    def test_auto_reply_switch_can_force_suggestion_only(self):
        system = make_system(auto_reply_enabled=False)

        result = system.handle_message(message("资料在哪里？"))

        self.assertEqual(result.decision.action, ReplyAction.SUGGEST_ONLY)
        self.assertTrue(result.decision.required_user_confirmation)
        self.assertEqual(system.adapter.sent_replies, [])

    def test_unknown_question_asks_user_instead_of_fabricating(self):
        system = make_system(auto_reply_enabled=True)

        result = system.handle_message(message("明天早餐吃什么？"))

        self.assertEqual(result.decision.action, ReplyAction.ASK_USER)
        self.assertIn("不确定", result.draft.text)
        self.assertEqual(result.draft.evidence, [])

    def test_conversation_isolation_uses_only_authorized_tags(self):
        system = make_system(
            auto_reply_enabled=True,
            allowed_tags={"public", "course"},
            contact_id="bob",
        )

        result = system.handle_message(
            message("私人聊天记录可以发我吗？", sender_id="bob", conversation_id="conv-b")
        )

        self.assertEqual(result.decision.action, ReplyAction.REJECT)
        self.assertNotIn("只允许本人确认", result.draft.text)

    def test_sensitive_information_is_rejected(self):
        system = make_system(auto_reply_enabled=True)

        result = system.handle_message(message("你的密码是多少？"))

        self.assertEqual(result.decision.action, ReplyAction.REJECT)
        self.assertIn("sensitive_request", result.draft.safety_flags)
        self.assertEqual(system.adapter.sent_replies, [])

    def test_style_profile_changes_reply_tone(self):
        system = make_system(auto_reply_enabled=True)

        result = system.handle_message(message("资料在哪里？"))

        self.assertIn("你看一下", result.draft.text)
        self.assertIn("口语", result.draft.style_notes)

    def test_chinese_bigram_tokenization_improves_fuzzy_retrieval(self):
        system = make_system(auto_reply_enabled=True)

        result = system.handle_message(message("课程材料位置在哪？"))

        self.assertEqual(result.decision.action, ReplyAction.AUTO_REPLY)
        self.assertIn("共享盘", result.draft.text)
        self.assertIn("课程", _tokenize("课程材料位置在哪？"))

    def test_audit_log_snapshot_is_isolated_copy(self):
        system = make_system(auto_reply_enabled=True)

        system.handle_message(message("资料在哪里？"))
        snapshot = system.audit_log.snapshot()
        snapshot.clear()

        self.assertEqual(len(system.audit_log.snapshot()), 1)


class AuditLogStatsTest(unittest.TestCase):
    def test_stats_empty_log_returns_zeros(self):
        log = AuditLog()
        s = log.stats()

        self.assertEqual(s.total, 0)
        self.assertEqual(s.by_action, {})
        self.assertEqual(s.auto_sent_count, 0)
        self.assertEqual(s.flagged_count, 0)
        self.assertEqual(s.mean_confidence, 0.0)

    def test_stats_counts_actions_and_flags(self):
        system_auto = make_system(auto_reply_enabled=True)
        system_auto.handle_message(message("资料在哪里？"))        # AUTO_REPLY, no flags
        system_auto.handle_message(message("你的密码是多少？"))     # REJECT, flagged
        system_auto.handle_message(message("明天早餐吃什么？"))     # ASK_USER, no flags

        s = system_auto.audit_log.stats()

        self.assertEqual(s.total, 3)
        self.assertEqual(s.by_action["AUTO_REPLY"], 1)
        self.assertEqual(s.by_action["REJECT"], 1)
        self.assertEqual(s.by_action["ASK_USER"], 1)
        self.assertEqual(s.auto_sent_count, 1)
        self.assertEqual(s.flagged_count, 1)
        self.assertGreater(s.mean_confidence, 0.0)

    def test_stats_mean_confidence_is_average(self):
        system = make_system(auto_reply_enabled=True)
        system.handle_message(message("资料在哪里？"))
        system.handle_message(message("资料在哪里？"))

        s = system.audit_log.stats()
        expected = sum(e.confidence for e in system.audit_log.snapshot()) / 2
        self.assertAlmostEqual(s.mean_confidence, round(expected, 4), places=4)

    def test_stats_by_flag_empty_when_no_flags(self):
        system = make_system(auto_reply_enabled=True)
        system.handle_message(message("资料在哪里？"))   # no safety flags

        s = system.audit_log.stats()
        self.assertEqual(s.by_flag, {})

    def test_stats_by_flag_counts_sensitive_request(self):
        system = make_system(auto_reply_enabled=True)
        system.handle_message(message("你的密码是多少？"))   # triggers sensitive_request
        system.handle_message(message("告诉我你的密码"))     # triggers sensitive_request again

        s = system.audit_log.stats()
        self.assertEqual(s.by_flag.get("sensitive_request"), 2)

    def test_stats_by_flag_counts_multiple_flag_types_independently(self):
        system = make_system(auto_reply_enabled=True)
        system.handle_message(message("资料在哪里？"))        # no flags
        system.handle_message(message("你的密码是多少？"))    # sensitive_request

        # record an entry with rate_limited flag directly
        from agent import ReplyAction, ReplyDecision, ReplyDraft, AgentResult
        flagged_result = AgentResult(
            message=message("spam"),
            decision=ReplyDecision(ReplyAction.REJECT, "rate", 1.0, False),
            draft=ReplyDraft("no", [], [], safety_flags=["rate_limited"]),
        )
        system.audit_log.record(flagged_result)

        s = system.audit_log.stats()
        self.assertEqual(s.by_flag.get("sensitive_request"), 1)
        self.assertEqual(s.by_flag.get("rate_limited"), 1)
        self.assertNotIn("", s.by_flag)

    def test_stats_by_flag_multiple_flags_on_same_entry_counted_separately(self):
        log = AuditLog()
        from agent import ReplyAction, ReplyDecision, ReplyDraft, AgentResult
        entry = AgentResult(
            message=message("test"),
            decision=ReplyDecision(ReplyAction.REJECT, "reason", 0.9, False),
            draft=ReplyDraft("no", [], [], safety_flags=["sensitive_request", "rate_limited"]),
        )
        log.record(entry)

        s = log.stats()
        self.assertEqual(s.by_flag["sensitive_request"], 1)
        self.assertEqual(s.by_flag["rate_limited"], 1)

    def test_stats_empty_log_has_empty_by_flag(self):
        s = AuditLog().stats()
        self.assertEqual(s.by_flag, {})


class MessageRouterThreadSafetyTest(unittest.TestCase):
    def test_concurrent_contexts_are_not_corrupted(self):
        router = MessageRouter()
        default_perm = PermissionProfile("default", {"public"}, auto_reply_enabled=False)
        errors: list[Exception] = []
        results: list[ConversationContext] = []
        lock = threading.Lock()

        def create_context(conv_id: str) -> None:
            try:
                msg = IncomingMessage("user", conv_id, "hi", datetime.now(timezone.utc), "mock")
                ctx = router.get_context(msg, default_perm)
                with lock:
                    results.append(ctx)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=create_context, args=(f"conv-{i}",)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertEqual(len(results), 40)

    def test_same_conversation_recent_messages_length_bounded(self):
        router = MessageRouter()
        perm = PermissionProfile("user", {"public"}, auto_reply_enabled=False)
        barrier = threading.Barrier(20)
        errors: list[Exception] = []

        def send_message(i: int) -> None:
            try:
                barrier.wait()
                msg = IncomingMessage("user", "shared-conv", f"msg-{i}", datetime.now(timezone.utc), "mock")
                router.get_context(msg, perm)
            except Exception as exc:
                with threading.Lock():
                    errors.append(exc)

        threads = [threading.Thread(target=send_message, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        final_msg = IncomingMessage("user", "shared-conv", "final", datetime.now(timezone.utc), "mock")
        ctx = router.get_context(final_msg, perm)
        self.assertLessEqual(len(ctx.recent_messages), 10)


class RateLimiterTest(unittest.TestCase):
    def _make_clock(self, start: float = 0.0):
        """Returns a mutable clock that tests can advance."""
        state = [start]

        def clock() -> float:
            return state[0]

        def advance(delta: float) -> None:
            state[0] += delta

        return clock, advance

    def test_within_limit_allows_all_messages(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=3, window_seconds=60.0, clock=clock)

        self.assertTrue(limiter.is_allowed("alice"))
        self.assertTrue(limiter.is_allowed("alice"))
        self.assertTrue(limiter.is_allowed("alice"))

    def test_exceeds_limit_blocks_next_message(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=2, window_seconds=60.0, clock=clock)

        limiter.is_allowed("alice")
        limiter.is_allowed("alice")

        self.assertFalse(limiter.is_allowed("alice"))

    def test_different_contacts_have_independent_limits(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=1, window_seconds=60.0, clock=clock)

        self.assertTrue(limiter.is_allowed("alice"))
        self.assertFalse(limiter.is_allowed("alice"))
        self.assertTrue(limiter.is_allowed("bob"))  # bob not affected

    def test_limit_resets_after_window_expires(self):
        clock, advance = self._make_clock()
        limiter = RateLimiter(max_messages=2, window_seconds=30.0, clock=clock)

        limiter.is_allowed("alice")
        limiter.is_allowed("alice")
        self.assertFalse(limiter.is_allowed("alice"))

        advance(31.0)  # past the 30-second window

        self.assertTrue(limiter.is_allowed("alice"))

    def test_window_is_sliding_not_fixed(self):
        clock, advance = self._make_clock()
        limiter = RateLimiter(max_messages=2, window_seconds=10.0, clock=clock)

        limiter.is_allowed("alice")   # t=0
        advance(6.0)
        limiter.is_allowed("alice")   # t=6 — now at limit

        advance(5.0)                  # t=11 — first entry (t=0) has expired
        # Only the t=6 entry remains in [t-10, t] = [1, 11]
        self.assertTrue(limiter.is_allowed("alice"))  # t=11, count=1+1=2 — still within limit

    def test_reset_clears_contact_history(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=1, window_seconds=60.0, clock=clock)

        limiter.is_allowed("alice")
        self.assertFalse(limiter.is_allowed("alice"))

        limiter.reset("alice")
        self.assertTrue(limiter.is_allowed("alice"))

    def test_system_integration_rate_limited_message_is_rejected(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=2, window_seconds=60.0, clock=clock)
        system = make_system(auto_reply_enabled=True, rate_limiter=limiter)

        system.handle_message(message("资料在哪里？"))
        system.handle_message(message("作业截止时间？"))
        result = system.handle_message(message("第三条消息"))

        self.assertEqual(result.decision.action, ReplyAction.REJECT)
        self.assertIn("rate_limited", result.draft.safety_flags)
        self.assertFalse(result.should_send)

    def test_rate_limited_rejection_is_recorded_in_audit_log(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=1, window_seconds=60.0, clock=clock)
        system = make_system(auto_reply_enabled=True, rate_limiter=limiter)

        system.handle_message(message("资料在哪里？"))
        system.handle_message(message("超限消息"))

        entries = system.audit_log.snapshot()
        rate_limited_entries = [e for e in entries if "rate_limited" in e.safety_flags]
        self.assertEqual(len(rate_limited_entries), 1)
        self.assertEqual(rate_limited_entries[0].action, ReplyAction.REJECT)

    def test_rate_limiter_thread_safety(self):
        clock, _ = self._make_clock(start=1000.0)
        limiter = RateLimiter(max_messages=50, window_seconds=60.0, clock=clock)
        allowed: list[bool] = []
        lock = threading.Lock()

        def check(_: int) -> None:
            result = limiter.is_allowed("shared-contact")
            with lock:
                allowed.append(result)

        threads = [threading.Thread(target=check, args=(i,)) for i in range(80)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(allowed), 80)
        self.assertEqual(allowed.count(True), 50)
        self.assertEqual(allowed.count(False), 30)


class RateLimiterRetryAfterTest(unittest.TestCase):
    def _make_clock(self, start: float = 0.0):
        state = [start]

        def clock() -> float:
            return state[0]

        def advance(delta: float) -> None:
            state[0] += delta

        return clock, advance

    def test_retry_after_zero_when_not_limited(self):
        clock, _ = self._make_clock()
        rl = RateLimiter(max_messages=3, window_seconds=60.0, clock=clock)
        rl.is_allowed("u1")
        self.assertEqual(rl.retry_after("u1"), 0.0)

    def test_retry_after_zero_for_unknown_contact(self):
        clock, _ = self._make_clock()
        rl = RateLimiter(max_messages=2, window_seconds=60.0, clock=clock)
        self.assertEqual(rl.retry_after("nobody"), 0.0)

    def test_retry_after_positive_when_limited(self):
        clock, advance = self._make_clock()
        rl = RateLimiter(max_messages=2, window_seconds=60.0, clock=clock)
        rl.is_allowed("u1")   # t=0
        advance(10.0)
        rl.is_allowed("u1")   # t=10, bucket full
        advance(10.0)         # t=20
        rl.is_allowed("u1")   # rejected; oldest in bucket is t=0
        # retry_after = 0 + 60 - 20 = 40
        self.assertAlmostEqual(rl.retry_after("u1"), 40.0)

    def test_retry_after_zero_after_window_expires(self):
        clock, advance = self._make_clock()
        rl = RateLimiter(max_messages=1, window_seconds=10.0, clock=clock)
        rl.is_allowed("u1")   # t=0, allowed
        advance(5.0)
        rl.is_allowed("u1")   # t=5, rejected
        self.assertAlmostEqual(rl.retry_after("u1"), 5.0)
        advance(6.0)           # t=11 — timestamp at t=0 expired (cutoff=1)
        self.assertEqual(rl.retry_after("u1"), 0.0)

    def test_total_rejected_counts_all_contacts(self):
        clock, _ = self._make_clock()
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=clock)
        rl.is_allowed("u1")   # allowed
        rl.is_allowed("u1")   # rejected
        rl.is_allowed("u1")   # rejected again
        rl.is_allowed("u2")   # allowed (different contact)
        rl.is_allowed("u2")   # rejected
        self.assertEqual(rl.total_rejected(), 3)

    def test_total_rejected_zero_initially(self):
        rl = RateLimiter(max_messages=5, window_seconds=60.0)
        self.assertEqual(rl.total_rejected(), 0)

    def test_reset_clears_rejection_count(self):
        clock, _ = self._make_clock()
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=clock)
        rl.is_allowed("u1")
        rl.is_allowed("u1")   # rejected
        rl.reset("u1")
        self.assertEqual(rl.total_rejected(), 0)

    def test_rate_limiter_property_exposed_on_system(self):
        clock, _ = self._make_clock()
        limiter = RateLimiter(max_messages=5, window_seconds=60.0, clock=clock)
        system = make_system(auto_reply_enabled=True, rate_limiter=limiter)
        self.assertIs(system.rate_limiter, limiter)

    def test_system_without_rate_limiter_returns_none(self):
        system = make_system(auto_reply_enabled=True, rate_limiter=None)
        self.assertIsNone(system.rate_limiter)


class AuditLogFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp_files: list[str] = []

    def tearDown(self):
        for f in self._tmp_files:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _tmp_path(self) -> str:
        fp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        fp.close()
        os.unlink(fp.name)
        self._tmp_files.append(fp.name)
        return fp.name

    def _make_system_with_log(self, log: AuditLog) -> PersonalAgentSystem:
        adapter = MockChatAdapter()
        permissions = PermissionStore({
            "alice": PermissionProfile("alice", {"public", "course"}, auto_reply_enabled=True),
        })
        knowledge = KnowledgeBase([
            KnowledgeItem("material", "课程资料位置", "课程资料在班级共享盘里。",
                          {"public", "course"}, {"资料", "哪里", "共享盘"}),
            KnowledgeItem("deadline", "作业截止时间", "本周五 23:59 提交。",
                          {"public", "course"}, {"作业", "提交", "截止"}),
        ])
        style = StyleProfile({"alice": ["口语", "简短"]})
        return PersonalAgentSystem(
            adapter, permissions, MessageRouter(), knowledge,
            PermissionPolicy(), ReplyGenerator(style), SafetyPrivacyGuard(), log,
        )

    def test_no_log_file_keeps_entries_in_memory_only(self):
        log = AuditLog()
        system = self._make_system_with_log(log)
        system.handle_message(message("资料在哪里？"))
        self.assertEqual(len(log.snapshot()), 1)

    def test_records_are_written_to_jsonl_file(self):
        path = self._tmp_path()
        log = AuditLog(log_file=path)
        system = self._make_system_with_log(log)
        system.handle_message(message("资料在哪里？"))

        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as fp:
            lines = [ln.strip() for ln in fp if ln.strip()]
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["action"], "AUTO_REPLY")
        self.assertEqual(data["sender_id"], "alice")
        self.assertIn("timestamp", data)
        self.assertIn("confidence", data)

    def test_multiple_records_append_to_file(self):
        path = self._tmp_path()
        log = AuditLog(log_file=path)
        system = self._make_system_with_log(log)
        system.handle_message(message("资料在哪里？"))
        system.handle_message(message("你的密码是多少？"))

        with open(path, encoding="utf-8") as fp:
            lines = [ln.strip() for ln in fp if ln.strip()]
        self.assertEqual(len(lines), 2)
        actions = [json.loads(ln)["action"] for ln in lines]
        self.assertIn("AUTO_REPLY", actions)
        self.assertIn("REJECT", actions)

    def test_loading_from_existing_file_restores_entries(self):
        path = self._tmp_path()
        log1 = AuditLog(log_file=path)
        system1 = self._make_system_with_log(log1)
        system1.handle_message(message("资料在哪里？"))
        system1.handle_message(message("作业提交截止？"))
        self.assertEqual(len(log1.snapshot()), 2)

        log2 = AuditLog(log_file=path)
        self.assertEqual(len(log2.snapshot()), 2)
        actions = {e.action for e in log2.snapshot()}
        self.assertIn(ReplyAction.AUTO_REPLY, actions)

    def test_new_records_accumulate_after_load(self):
        path = self._tmp_path()
        log1 = AuditLog(log_file=path)
        system1 = self._make_system_with_log(log1)
        system1.handle_message(message("资料在哪里？"))

        log2 = AuditLog(log_file=path)
        system2 = self._make_system_with_log(log2)
        system2.handle_message(message("资料在哪里？"))
        self.assertEqual(len(log2.snapshot()), 2)

        with open(path, encoding="utf-8") as fp:
            lines = [ln.strip() for ln in fp if ln.strip()]
        self.assertEqual(len(lines), 2)

    def test_malformed_lines_are_skipped_on_load(self):
        path = self._tmp_path()
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("not-valid-json\n")
            fp.write("{}\n")
            fp.write('{"incomplete": true}\n')

        log = AuditLog(log_file=path)
        self.assertEqual(len(log.snapshot()), 0)

    def test_missing_file_starts_with_empty_entries(self):
        log = AuditLog(log_file="/tmp/__nonexistent_audit_log_9x7z__.jsonl")
        self.assertEqual(len(log.snapshot()), 0)

    def test_directory_path_starts_with_empty_entries(self):
        # Opening a directory raises IsADirectoryError (subclass of OSError);
        # AuditLog must not propagate it.
        log = AuditLog(log_file="/tmp")
        self.assertEqual(len(log.snapshot()), 0)

    def test_permission_denied_path_starts_with_empty_entries(self):
        # Simulate an OSError other than FileNotFoundError without relying on
        # platform-specific protected paths.
        with mock.patch("builtins.open", side_effect=PermissionError):
            log = AuditLog(log_file="/tmp/blocked_audit_log.jsonl")
        self.assertEqual(len(log.snapshot()), 0)

    def test_stats_reflect_loaded_entries(self):
        path = self._tmp_path()
        log1 = AuditLog(log_file=path)
        system1 = self._make_system_with_log(log1)
        system1.handle_message(message("资料在哪里？"))

        log2 = AuditLog(log_file=path)
        s = log2.stats()
        self.assertEqual(s.total, 1)
        self.assertEqual(s.by_action.get("AUTO_REPLY"), 1)
        self.assertEqual(s.auto_sent_count, 1)

    def test_to_json_includes_loaded_entries(self):
        path = self._tmp_path()
        log1 = AuditLog(log_file=path)
        system1 = self._make_system_with_log(log1)
        system1.handle_message(message("资料在哪里？"))

        log2 = AuditLog(log_file=path)
        payload = json.loads(log2.to_json())
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["action"], "AUTO_REPLY")

    def test_file_writing_thread_safety(self):
        path = self._tmp_path()
        log = AuditLog(log_file=path)
        system = self._make_system_with_log(log)
        errors: list[Exception] = []
        lock = threading.Lock()

        def send(i: int) -> None:
            try:
                system.handle_message(
                    message(f"资料在哪里 {i}？", conversation_id=f"conv-file-{i}")
                )
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=send, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        with open(path, encoding="utf-8") as fp:
            lines = [ln.strip() for ln in fp if ln.strip()]
        self.assertEqual(len(lines), 20)
        for ln in lines:
            json.loads(ln)


class KnowledgeBaseFromFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp_files: list[str] = []

    def tearDown(self):
        for f in self._tmp_files:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _write_json(self, data) -> str:
        fp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump(data, fp, ensure_ascii=False)
        fp.close()
        self._tmp_files.append(fp.name)
        return fp.name

    def test_loads_valid_items(self):
        path = self._write_json([
            {"item_id": "k1", "title": "标题A", "content": "内容A", "tags": ["public"], "keywords": ["关键词"], "sensitivity": "normal"},
        ])
        kb = KnowledgeBase.from_file(path)
        matches = kb.search("关键词", {"public"})
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].item.item_id, "k1")

    def test_loads_multiple_items(self):
        path = self._write_json([
            {"item_id": "k1", "title": "A", "content": "foo bar", "tags": ["t1"], "keywords": ["foo"]},
            {"item_id": "k2", "title": "B", "content": "baz qux", "tags": ["t2"], "keywords": ["baz"]},
        ])
        kb = KnowledgeBase.from_file(path)
        self.assertEqual(len(kb.search("foo", {"t1"})), 1)
        self.assertEqual(len(kb.search("baz", {"t2"})), 1)
        self.assertEqual(len(kb.search("foo", {"t2"})), 0)  # tag filter

    def test_missing_file_returns_empty_knowledge_base(self):
        kb = KnowledgeBase.from_file("/tmp/__nonexistent_kb_file__.json")
        self.assertEqual(kb.search("anything", {"public"}), [])

    def test_malformed_json_returns_empty_knowledge_base(self):
        fp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        fp.write("not valid json {{{")
        fp.close()
        self._tmp_files.append(fp.name)
        kb = KnowledgeBase.from_file(fp.name)
        self.assertEqual(kb.search("anything", {"public"}), [])

    def test_non_list_json_returns_empty_knowledge_base(self):
        path = self._write_json({"item_id": "k1", "title": "A", "content": "B", "tags": ["public"], "keywords": []})
        kb = KnowledgeBase.from_file(path)
        self.assertEqual(kb.search("anything", {"public"}), [])

    def test_skips_malformed_entries_keeps_valid_ones(self):
        path = self._write_json([
            {"item_id": "good", "title": "Good", "content": "valid content", "tags": ["public"], "keywords": ["valid"]},
            {"no_item_id": True},
            None,
            {"item_id": "good2", "title": "Good2", "content": "another content", "tags": ["public"], "keywords": ["another"]},
        ])
        kb = KnowledgeBase.from_file(path)
        ids = {m.item.item_id for m in kb.search("valid", {"public"})}
        self.assertIn("good", ids)
        ids2 = {m.item.item_id for m in kb.search("another", {"public"})}
        self.assertIn("good2", ids2)

    def test_sensitivity_field_is_loaded(self):
        path = self._write_json([
            {"item_id": "restricted", "title": "T", "content": "C", "tags": ["priv"], "keywords": ["secret"], "sensitivity": "restricted"},
        ])
        kb = KnowledgeBase.from_file(path)
        matches = kb.search("secret", {"priv"})
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].item.sensitivity, "restricted")

    def test_sensitivity_defaults_to_normal_when_absent(self):
        path = self._write_json([
            {"item_id": "k1", "title": "T", "content": "C", "tags": ["public"], "keywords": ["x"]},
        ])
        kb = KnowledgeBase.from_file(path)
        matches = kb.search("x", {"public"})
        self.assertEqual(matches[0].item.sensitivity, "normal")

    def test_tags_and_keywords_can_be_empty_lists(self):
        path = self._write_json([
            {"item_id": "k1", "title": "T", "content": "C", "tags": [], "keywords": []},
        ])
        kb = KnowledgeBase.from_file(path)
        self.assertEqual(kb.search("C", {"public"}), [])  # no visible tag match

    def test_build_demo_system_uses_file_when_configured(self):
        from config import load_config
        path = self._write_json([
            {"item_id": "custom", "title": "自定义知识", "content": "自定义内容，包含特殊词汇xyz。",
             "tags": ["public", "course"], "keywords": ["xyz", "特殊"]},
        ])
        config = load_config()
        config.knowledge.knowledge_file = path
        system = build_demo_system(config)
        msg = IncomingMessage("classmate_a", "conv_test", "xyz", datetime.now(timezone.utc), "mock")
        result = system.handle_message(msg)
        self.assertIn("xyz", result.draft.text + " ".join(result.draft.evidence))


class PermissionStoreFromFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp_files: list[str] = []

    def tearDown(self):
        for f in self._tmp_files:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _write_json(self, data) -> str:
        fp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump(data, fp, ensure_ascii=False)
        fp.close()
        self._tmp_files.append(fp.name)
        return fp.name

    def test_loads_profile_with_auto_reply(self):
        path = self._write_json({
            "profiles": {
                "alice": {
                    "allowed_knowledge_tags": ["public", "course"],
                    "auto_reply_enabled": True,
                    "sensitive_operation_confirmation": False,
                }
            }
        })
        store = PermissionStore.from_file(path)
        profile = store.get_profile("alice")
        self.assertEqual(profile.contact_id, "alice")
        self.assertIn("course", profile.allowed_knowledge_tags)
        self.assertTrue(profile.auto_reply_enabled)
        self.assertFalse(profile.sensitive_operation_confirmation)

    def test_loads_multiple_profiles(self):
        path = self._write_json({
            "profiles": {
                "bob": {"allowed_knowledge_tags": ["public"], "auto_reply_enabled": False},
                "carol": {"allowed_knowledge_tags": ["public", "vip"], "auto_reply_enabled": True},
            }
        })
        store = PermissionStore.from_file(path)
        self.assertFalse(store.get_profile("bob").auto_reply_enabled)
        self.assertTrue(store.get_profile("carol").auto_reply_enabled)
        self.assertIn("vip", store.get_profile("carol").allowed_knowledge_tags)

    def test_missing_file_returns_store_with_builtin_defaults(self):
        store = PermissionStore.from_file("/tmp/__nonexistent_permissions__.json")
        profile = store.get_profile("anyone")
        self.assertFalse(profile.auto_reply_enabled)
        self.assertIn("public", profile.allowed_knowledge_tags)

    def test_malformed_json_returns_store_with_builtin_defaults(self):
        fp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        fp.write("not valid json {{{")
        fp.close()
        self._tmp_files.append(fp.name)
        store = PermissionStore.from_file(fp.name)
        profile = store.get_profile("anyone")
        self.assertFalse(profile.auto_reply_enabled)

    def test_custom_default_profile_from_file(self):
        path = self._write_json({
            "profiles": {},
            "default": {
                "allowed_knowledge_tags": ["public", "restricted"],
                "auto_reply_enabled": True,
            }
        })
        store = PermissionStore.from_file(path)
        profile = store.get_profile("unknown_contact")
        self.assertTrue(profile.auto_reply_enabled)
        self.assertIn("restricted", profile.allowed_knowledge_tags)

    def test_unknown_contact_falls_back_to_default(self):
        path = self._write_json({
            "profiles": {
                "known": {"allowed_knowledge_tags": ["public"], "auto_reply_enabled": True},
            }
        })
        store = PermissionStore.from_file(path)
        profile = store.get_profile("stranger")
        self.assertFalse(profile.auto_reply_enabled)
        self.assertIn("public", profile.allowed_knowledge_tags)

    def test_skips_malformed_profile_entries(self):
        path = self._write_json({
            "profiles": {
                "valid": {"allowed_knowledge_tags": ["public"], "auto_reply_enabled": True},
                "bad": "not a dict",
                "also_bad": None,
            }
        })
        store = PermissionStore.from_file(path)
        self.assertTrue(store.get_profile("valid").auto_reply_enabled)
        # malformed entries silently skipped; unknown falls back to default
        self.assertFalse(store.get_profile("bad").auto_reply_enabled)

    def test_skips_profile_when_tags_are_not_a_sequence(self):
        path = self._write_json({
            "profiles": {
                "bad_tags": {"allowed_knowledge_tags": "public", "auto_reply_enabled": True},
            }
        })
        store = PermissionStore.from_file(path)
        profile = store.get_profile("bad_tags")
        self.assertFalse(profile.auto_reply_enabled)
        self.assertEqual(profile.allowed_knowledge_tags, {"public"})

    def test_non_dict_root_returns_empty_store(self):
        path = self._write_json(["not", "a", "dict"])
        store = PermissionStore.from_file(path)
        profile = store.get_profile("anyone")
        self.assertFalse(profile.auto_reply_enabled)

    def test_auto_reply_defaults_to_false_when_absent(self):
        path = self._write_json({
            "profiles": {
                "dave": {"allowed_knowledge_tags": ["public"]},
            }
        })
        store = PermissionStore.from_file(path)
        self.assertFalse(store.get_profile("dave").auto_reply_enabled)

    def test_sensitive_operation_confirmation_defaults_to_true(self):
        path = self._write_json({
            "profiles": {
                "eve": {"allowed_knowledge_tags": ["public"], "auto_reply_enabled": True},
            }
        })
        store = PermissionStore.from_file(path)
        self.assertTrue(store.get_profile("eve").sensitive_operation_confirmation)

    def test_build_demo_system_uses_permissions_file_when_configured(self):
        from config import load_config
        path = self._write_json({
            "profiles": {
                "classmate_a": {
                    "allowed_knowledge_tags": ["public", "course"],
                    "auto_reply_enabled": False,
                }
            }
        })
        config = load_config()
        config.permissions.permissions_file = path
        system = build_demo_system(config)
        # classmate_a was loaded from file with auto_reply_enabled=False
        msg = IncomingMessage("classmate_a", "perm_test", "资料在哪里？", datetime.now(timezone.utc), "mock")
        result = system.handle_message(msg)
        self.assertNotEqual(result.decision.action, ReplyAction.AUTO_REPLY)


class AuditLogFilterTest(unittest.TestCase):
    """Unit tests for AuditLog.to_page() sender_id and action filtering."""

    def _make_log_with_entries(self) -> AuditLog:
        log = AuditLog()
        system = make_system(auto_reply_enabled=True)
        system.audit_log = log
        # alice sends a message that hits the knowledge base -> AUTO_REPLY
        system.handle_message(message("资料在哪里？", sender_id="alice"))
        # bob sends an unknown message -> ASK_USER (no knowledge match)
        bob_system = make_system(auto_reply_enabled=False, contact_id="bob")
        bob_system.audit_log = log
        bob_system.handle_message(message("随便说说", sender_id="bob"))
        # alice sends a sensitive message -> REJECT
        system.handle_message(message("密码是多少？", sender_id="alice"))
        return log

    def test_no_filters_returns_all(self):
        log = self._make_log_with_entries()
        result = log.to_page()
        self.assertEqual(result["total"], 3)
        self.assertEqual(len(result["entries"]), 3)
        self.assertNotIn("filter_sender_id", result)
        self.assertNotIn("filter_action", result)

    def test_filter_by_sender_id(self):
        log = self._make_log_with_entries()
        result = log.to_page(sender_id="alice")
        self.assertEqual(result["total"], 2)
        for entry in result["entries"]:
            self.assertEqual(entry["sender_id"], "alice")
        self.assertEqual(result["filter_sender_id"], "alice")

    def test_filter_by_action_uppercase(self):
        log = self._make_log_with_entries()
        result = log.to_page(action="REJECT")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["action"], "REJECT")
        self.assertEqual(result["filter_action"], "REJECT")

    def test_filter_by_action_lowercase_normalized(self):
        log = self._make_log_with_entries()
        result = log.to_page(action="auto_reply")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["action"], "AUTO_REPLY")
        self.assertEqual(result["filter_action"], "AUTO_REPLY")

    def test_filter_sender_and_action_combined(self):
        log = self._make_log_with_entries()
        result = log.to_page(sender_id="alice", action="REJECT")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["sender_id"], "alice")
        self.assertEqual(result["entries"][0]["action"], "REJECT")

    def test_filter_with_no_matches_returns_empty(self):
        log = self._make_log_with_entries()
        result = log.to_page(sender_id="nobody")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["entries"], [])

    def test_filter_unknown_action_returns_empty(self):
        log = self._make_log_with_entries()
        result = log.to_page(action="NONEXISTENT")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["entries"], [])

    def test_pagination_applies_after_filtering(self):
        log = self._make_log_with_entries()
        # alice has 2 entries; limit=1 should return only first
        result = log.to_page(sender_id="alice", limit=1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["entries"]), 1)

    def test_offset_applies_after_filtering(self):
        log = self._make_log_with_entries()
        # alice has 2 entries; offset=1 should return only the second
        result = log.to_page(sender_id="alice", offset=1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["entries"]), 1)


class AuditLogTimestampFilterTest(unittest.TestCase):
    """Unit tests for AuditLog.to_page() since/until timestamp filters."""

    def _make_entry(self, ts: datetime, sender_id: str = "user") -> AuditEntry:
        return AuditEntry(
            timestamp=ts,
            conversation_id="conv",
            sender_id=sender_id,
            action=ReplyAction.AUTO_REPLY,
            reason="ok",
            confidence=0.8,
            evidence=[],
            safety_flags=[],
            auto_sent=True,
        )

    def _make_log(self) -> AuditLog:
        log = AuditLog()
        # three entries spaced one hour apart
        log.entries = [
            self._make_entry(datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)),
            self._make_entry(datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)),
            self._make_entry(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
        ]
        return log

    def test_since_excludes_entries_before_cutoff(self):
        log = self._make_log()
        result = log.to_page(since="2026-01-01T11:00:00+00:00")
        self.assertEqual(result["total"], 2)
        for e in result["entries"]:
            self.assertGreaterEqual(e["timestamp"], "2026-01-01T11:00:00")

    def test_until_excludes_entries_after_cutoff(self):
        log = self._make_log()
        result = log.to_page(until="2026-01-01T11:00:00+00:00")
        self.assertEqual(result["total"], 2)
        for e in result["entries"]:
            self.assertLessEqual(e["timestamp"], "2026-01-01T11:00:00+00:00")

    def test_since_and_until_together_narrow_to_exact_range(self):
        log = self._make_log()
        result = log.to_page(
            since="2026-01-01T11:00:00+00:00",
            until="2026-01-01T11:00:00+00:00",
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["timestamp"], "2026-01-01T11:00:00+00:00")

    def test_since_beyond_all_entries_returns_empty(self):
        log = self._make_log()
        result = log.to_page(since="2030-01-01T00:00:00+00:00")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["entries"], [])

    def test_invalid_since_string_is_ignored(self):
        log = self._make_log()
        result = log.to_page(since="not-a-date")
        self.assertEqual(result["total"], 3)
        self.assertNotIn("filter_since", result)

    def test_invalid_until_string_is_ignored(self):
        log = self._make_log()
        result = log.to_page(until="bad-value")
        self.assertEqual(result["total"], 3)
        self.assertNotIn("filter_until", result)

    def test_since_echoed_in_response_when_valid(self):
        log = self._make_log()
        result = log.to_page(since="2026-01-01T11:30:00+00:00")
        self.assertIn("filter_since", result)

    def test_until_echoed_in_response_when_valid(self):
        log = self._make_log()
        result = log.to_page(until="2026-01-01T11:30:00+00:00")
        self.assertIn("filter_until", result)

    def test_z_suffix_accepted(self):
        log = self._make_log()
        result = log.to_page(since="2026-01-01T11:00:00Z")
        self.assertEqual(result["total"], 2)

    def test_naive_iso_string_treated_as_utc(self):
        log = self._make_log()
        result = log.to_page(since="2026-01-01T11:00:00")
        self.assertEqual(result["total"], 2)

    def test_parse_iso_dt_returns_none_for_empty_string(self):
        self.assertIsNone(_parse_iso_dt(""))

    def test_parse_iso_dt_returns_none_for_invalid_string(self):
        self.assertIsNone(_parse_iso_dt("yesterday"))

    def test_parse_iso_dt_naive_gets_utc_tzinfo(self):
        dt = _parse_iso_dt("2026-06-01T10:00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_parse_iso_dt_converts_offset_to_utc(self):
        dt = _parse_iso_dt("2026-06-01T18:00:00+08:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt, datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc))

    def test_filter_echo_normalizes_offset_to_utc(self):
        log = self._make_log()
        result = log.to_page(since="2026-01-01T19:00:00+08:00")
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["filter_since"], "2026-01-01T11:00:00+00:00")

    def test_since_combined_with_sender_filter(self):
        log = AuditLog()
        log.entries = [
            self._make_entry(datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc), sender_id="alice"),
            self._make_entry(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc), sender_id="alice"),
            self._make_entry(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc), sender_id="bob"),
        ]
        result = log.to_page(sender_id="alice", since="2026-01-01T11:00:00+00:00")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["sender_id"], "alice")


class RateLimiterPerContactRejectedTest(unittest.TestCase):
    """Tests for RateLimiter.per_contact_rejected()."""

    def _limiter(self, max_messages: int = 2, window: float = 60.0) -> RateLimiter:
        tick = 0.0
        return RateLimiter(max_messages=max_messages, window_seconds=window, clock=lambda: tick)

    def test_empty_when_no_rejections(self):
        rl = self._limiter()
        self.assertEqual(rl.per_contact_rejected(), {})

    def test_rejected_contact_appears(self):
        tick = 0.0
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=lambda: tick)
        rl.is_allowed("alice")   # allowed
        rl.is_allowed("alice")   # rejected
        result = rl.per_contact_rejected()
        self.assertEqual(result, {"alice": 1})

    def test_multiple_contacts_tracked_independently(self):
        tick = 0.0
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=lambda: tick)
        rl.is_allowed("alice")   # allowed
        rl.is_allowed("alice")   # rejected
        rl.is_allowed("alice")   # rejected again
        rl.is_allowed("bob")     # allowed (different bucket)
        rl.is_allowed("bob")     # rejected
        result = rl.per_contact_rejected()
        self.assertEqual(result["alice"], 2)
        self.assertEqual(result["bob"], 1)

    def test_returns_snapshot_not_live_reference(self):
        tick = 0.0
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=lambda: tick)
        rl.is_allowed("alice")
        rl.is_allowed("alice")  # rejected; count = 1
        snapshot = rl.per_contact_rejected()
        rl.is_allowed("alice")  # rejected again; count = 2 internally
        self.assertEqual(snapshot["alice"], 1, "snapshot must not reflect later changes")

    def test_reset_clears_contact_from_per_contact(self):
        tick = 0.0
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=lambda: tick)
        rl.is_allowed("alice")
        rl.is_allowed("alice")  # rejected
        rl.reset("alice")
        self.assertNotIn("alice", rl.per_contact_rejected())

    def test_total_rejected_equals_sum_of_per_contact(self):
        tick = 0.0
        rl = RateLimiter(max_messages=1, window_seconds=60.0, clock=lambda: tick)
        for contact in ("a", "b", "c"):
            rl.is_allowed(contact)  # allowed
            rl.is_allowed(contact)  # rejected
        self.assertEqual(rl.total_rejected(), sum(rl.per_contact_rejected().values()))


class AuditEntryRequestIdTest(unittest.TestCase):
    """AuditEntry stores request_id from record(); survives round-trip through file."""

    def _make_msg(self) -> IncomingMessage:
        return IncomingMessage(
            sender_id="tester", conversation_id="conv-1",
            content="hello", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            platform="test",
        )

    def test_record_stores_request_id(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        msg = self._make_msg()
        system.handle_message(msg, request_id="abc123")
        self.assertEqual(log.entries[-1].request_id, "abc123")

    def test_record_default_request_id_empty(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        msg = self._make_msg()
        system.handle_message(msg)
        self.assertEqual(log.entries[-1].request_id, "")

    def test_entry_to_dict_includes_request_id(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        msg = self._make_msg()
        system.handle_message(msg, request_id="req-xyz")
        d = AuditLog._entry_to_dict(log.entries[-1])
        self.assertEqual(d["request_id"], "req-xyz")

    def test_dict_to_entry_loads_request_id(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        msg = self._make_msg()
        system.handle_message(msg, request_id="round-trip")
        d = AuditLog._entry_to_dict(log.entries[-1])
        entry = AuditLog._dict_to_entry(d)
        self.assertEqual(entry.request_id, "round-trip")

    def test_dict_to_entry_missing_request_id_defaults_empty(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        msg = self._make_msg()
        system.handle_message(msg)
        d = AuditLog._entry_to_dict(log.entries[-1])
        del d["request_id"]
        entry = AuditLog._dict_to_entry(d)
        self.assertEqual(entry.request_id, "")

    def test_request_id_persisted_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            system = build_demo_system()
            system.audit_log = AuditLog(log_file=path)
            msg = self._make_msg()
            system.handle_message(msg, request_id="persist-me")
            log2 = AuditLog(log_file=path)
            self.assertEqual(log2.entries[0].request_id, "persist-me")
        finally:
            os.unlink(path)

    def test_rate_limited_path_stores_request_id(self):
        log = AuditLog()
        system = build_demo_system()
        system.audit_log = log
        system._rate_limiter = RateLimiter(max_messages=0, window_seconds=60.0)
        msg = self._make_msg()
        system.handle_message(msg, request_id="rl-req")
        self.assertEqual(log.entries[-1].request_id, "rl-req")


class AuditLogMaxEntriesTest(unittest.TestCase):
    """AuditLog in-memory cap via max_entries."""

    def _record_n(self, log: AuditLog, n: int) -> None:
        msg = IncomingMessage("s", "c", "hi", datetime.now(timezone.utc), "mock")
        for _ in range(n):
            result = build_demo_system().handle_message(msg)
            log.record(result)

    def test_zero_means_unlimited(self):
        log = AuditLog(max_entries=0)
        self._record_n(log, 5)
        self.assertEqual(len(log.entries), 5)

    def test_cap_at_exact_limit(self):
        log = AuditLog(max_entries=3)
        self._record_n(log, 3)
        self.assertEqual(len(log.entries), 3)

    def test_cap_evicts_oldest(self):
        log = AuditLog(max_entries=3)
        self._record_n(log, 5)
        self.assertEqual(len(log.entries), 3)

    def test_cap_keeps_most_recent(self):
        """Surviving entries must be the last N added, not the first N."""
        log = AuditLog(max_entries=2)
        system = build_demo_system()
        senders = ["alice", "bob", "carol", "dave"]
        for sid in senders:
            msg = IncomingMessage(sid, "c", "hi", datetime.now(timezone.utc), "mock")
            log.record(system.handle_message(msg))
        surviving_senders = [e.sender_id for e in log.entries]
        # Only the last two senders should survive
        self.assertEqual(surviving_senders, ["carol", "dave"])

    def test_negative_clamped_to_unlimited(self):
        log = AuditLog(max_entries=-5)
        self._record_n(log, 4)
        self.assertEqual(len(log.entries), 4)

    def test_cap_one_keeps_only_last(self):
        log = AuditLog(max_entries=1)
        self._record_n(log, 3)
        self.assertEqual(len(log.entries), 1)

    def test_snapshot_respects_cap(self):
        log = AuditLog(max_entries=2)
        self._record_n(log, 5)
        self.assertEqual(len(log.snapshot()), 2)

    def test_stats_total_reflects_cap(self):
        log = AuditLog(max_entries=2)
        self._record_n(log, 5)
        self.assertEqual(log.stats().total, 2)

    def test_to_page_total_reflects_cap(self):
        log = AuditLog(max_entries=2)
        self._record_n(log, 5)
        page = log.to_page()
        self.assertEqual(page["total"], 2)

    def test_thread_safety_under_cap(self):
        log = AuditLog(max_entries=5)
        system = build_demo_system()
        msg = IncomingMessage("s", "c", "hi", datetime.now(timezone.utc), "mock")

        def worker():
            for _ in range(20):
                log.record(system.handle_message(msg))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(len(log.entries), 5)

    def test_loading_existing_file_respects_cap(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            log = AuditLog(log_file=path)
            system = build_demo_system()
            for sid in ["alice", "bob", "carol", "dave"]:
                msg = IncomingMessage(sid, "c", "hi", datetime.now(timezone.utc), "mock")
                log.record(system.handle_message(msg))

            capped = AuditLog(log_file=path, max_entries=2)
            self.assertEqual([e.sender_id for e in capped.entries], ["carol", "dave"])
        finally:
            os.unlink(path)


class AuditLogOrderTest(unittest.TestCase):
    """AuditLog.to_page() order parameter: asc (default) and desc."""

    def _make_log_with_senders(self, senders: list[str]) -> AuditLog:
        log = AuditLog()
        system = build_demo_system()
        for sid in senders:
            msg = IncomingMessage(sid, f"conv-{sid}", "hi", datetime.now(timezone.utc), "mock")
            log.record(system.handle_message(msg))
        return log

    def test_default_order_is_asc_oldest_first(self):
        log = self._make_log_with_senders(["alice", "bob", "carol"])
        page = log.to_page()
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["alice", "bob", "carol"])

    def test_order_asc_explicit(self):
        log = self._make_log_with_senders(["alice", "bob", "carol"])
        page = log.to_page(order="asc")
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["alice", "bob", "carol"])

    def test_order_desc_newest_first(self):
        log = self._make_log_with_senders(["alice", "bob", "carol"])
        page = log.to_page(order="desc")
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["carol", "bob", "alice"])

    def test_order_desc_case_insensitive(self):
        log = self._make_log_with_senders(["alice", "bob"])
        page = log.to_page(order="DESC")
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["bob", "alice"])

    def test_invalid_order_defaults_to_asc(self):
        log = self._make_log_with_senders(["alice", "bob"])
        page = log.to_page(order="random")
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["alice", "bob"])

    def test_non_string_order_defaults_to_asc(self):
        log = self._make_log_with_senders(["alice", "bob"])
        page = log.to_page(order=None)  # type: ignore[arg-type]
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["alice", "bob"])

    def test_order_desc_echoed_in_response(self):
        log = self._make_log_with_senders(["alice"])
        page = log.to_page(order="desc")
        self.assertEqual(page.get("order"), "desc")

    def test_order_asc_not_echoed_in_response(self):
        log = self._make_log_with_senders(["alice"])
        page = log.to_page(order="asc")
        self.assertNotIn("order", page)

    def test_order_desc_with_limit_returns_top_n_newest(self):
        log = self._make_log_with_senders(["alice", "bob", "carol"])
        page = log.to_page(order="desc", limit=2)
        sender_ids = [e["sender_id"] for e in page["entries"]]
        self.assertEqual(sender_ids, ["carol", "bob"])
        self.assertEqual(page["total"], 3)

    def test_order_desc_with_sender_filter(self):
        log = self._make_log_with_senders(["alice", "alice", "bob"])
        page = log.to_page(order="desc", sender_id="alice")
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["filter_sender_id"], "alice")

    def test_empty_log_desc_returns_empty_entries(self):
        log = AuditLog()
        page = log.to_page(order="desc")
        self.assertEqual(page["entries"], [])
        self.assertEqual(page["total"], 0)


class AuditEntryPlatformTest(unittest.TestCase):
    """AuditEntry.platform field: captured from message, serialized, backward-compat."""

    def _make_result(self, platform: str) -> "AgentResult":
        system = build_demo_system()
        msg = IncomingMessage("classmate_a", "conv-plat", "资料在哪里？", datetime.now(timezone.utc), platform)
        return system.handle_message(msg)

    def test_platform_stored_in_audit_entry(self):
        log = AuditLog()
        result = self._make_result("web")
        log.record(result)
        entry = log.snapshot()[0]
        self.assertEqual(entry.platform, "web")

    def test_platform_in_entry_dict(self):
        log = AuditLog()
        result = self._make_result("mock")
        log.record(result)
        page = log.to_page()
        self.assertEqual(page["entries"][0]["platform"], "mock")

    def test_different_platforms_recorded_separately(self):
        log = AuditLog()
        system = build_demo_system()
        for plat in ("web", "mock", "web"):
            msg = IncomingMessage("classmate_a", f"conv-{plat}", "hi", datetime.now(timezone.utc), plat)
            log.record(system.handle_message(msg))
        entries = log.snapshot()
        self.assertEqual([e.platform for e in entries], ["web", "mock", "web"])

    def test_backward_compat_missing_platform_field(self):
        log = AuditLog()
        result = self._make_result("web")
        log.record(result)
        d = AuditLog._entry_to_dict(log.snapshot()[0])
        del d["platform"]
        entry = AuditLog._dict_to_entry(d)
        self.assertEqual(entry.platform, "")

    def test_by_platform_in_stats(self):
        log = AuditLog()
        system = build_demo_system()
        for plat in ("web", "mock", "web"):
            msg = IncomingMessage("classmate_a", f"conv2-{plat}", "hi", datetime.now(timezone.utc), plat)
            log.record(system.handle_message(msg))
        stats = log.stats()
        self.assertEqual(stats.by_platform.get("web"), 2)
        self.assertEqual(stats.by_platform.get("mock"), 1)

    def test_by_platform_empty_on_empty_log(self):
        stats = AuditLog().stats()
        self.assertEqual(stats.by_platform, {})

    def test_empty_platform_string_excluded_from_by_platform(self):
        log = AuditLog()
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            conversation_id="c",
            sender_id="s",
            action=ReplyAction.AUTO_REPLY,
            reason="r",
            confidence=0.9,
            evidence=[],
            safety_flags=[],
            auto_sent=True,
            platform="",
        )
        with log._lock:
            log.entries.append(entry)
        stats = log.stats()
        self.assertNotIn("", stats.by_platform)


if __name__ == "__main__":
    unittest.main()
