from datetime import datetime, timezone
import unittest

from agent import (
    AuditLog,
    IncomingMessage,
    KnowledgeBase,
    KnowledgeItem,
    MessageRouter,
    MockChatAdapter,
    PermissionPolicy,
    PermissionProfile,
    PermissionStore,
    PersonalAgentSystem,
    ReplyAction,
    ReplyGenerator,
    SafetyPrivacyGuard,
    StyleProfile,
)


def make_system(auto_reply_enabled=True, allowed_tags=None, contact_id="alice"):
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


if __name__ == "__main__":
    unittest.main()
