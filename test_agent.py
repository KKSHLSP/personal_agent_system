from datetime import datetime, timezone
import json
import os
import tempfile
import threading
import unittest

from agent import (
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


if __name__ == "__main__":
    unittest.main()
