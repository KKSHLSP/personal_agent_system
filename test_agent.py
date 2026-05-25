from datetime import datetime, timezone
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


if __name__ == "__main__":
    unittest.main()
