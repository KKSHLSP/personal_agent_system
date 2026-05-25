"""Personal digital avatar agent for low-risk social communication tasks.

This module implements a small but complete architecture:

Platform Adapter -> Message Router -> Permission Policy -> Knowledge Retrieval
-> Reply Generator -> Safety & Privacy -> Audit Log

The first version uses an in-memory mock chat adapter and local Python data
structures so it can be demonstrated and tested without any external service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Protocol
import json
import re
import threading
import time

if TYPE_CHECKING:
    from config import AgentConfig


class ReplyAction(str, Enum):
    AUTO_REPLY = "AUTO_REPLY"
    SUGGEST_ONLY = "SUGGEST_ONLY"
    ASK_USER = "ASK_USER"
    REJECT = "REJECT"


@dataclass(frozen=True)
class IncomingMessage:
    sender_id: str
    conversation_id: str
    content: str
    timestamp: datetime
    platform: str


@dataclass(frozen=True)
class PermissionProfile:
    contact_id: str
    allowed_knowledge_tags: set[str]
    auto_reply_enabled: bool = False
    sensitive_operation_confirmation: bool = True


@dataclass(frozen=True)
class KnowledgeItem:
    item_id: str
    title: str
    content: str
    tags: set[str]
    keywords: set[str]
    sensitivity: str = "normal"


@dataclass(frozen=True)
class KnowledgeMatch:
    item: KnowledgeItem
    score: int
    matched_keywords: set[str]


@dataclass(frozen=True)
class ReplyDecision:
    action: ReplyAction
    reason: str
    confidence: float
    required_user_confirmation: bool


@dataclass(frozen=True)
class ReplyDraft:
    text: str
    evidence: list[str]
    style_notes: list[str]
    safety_flags: list[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    conversation_id: str
    contact_id: str
    visible_knowledge_tags: set[str]
    recent_messages: list[IncomingMessage] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResult:
    message: IncomingMessage
    decision: ReplyDecision
    draft: ReplyDraft

    @property
    def should_send(self) -> bool:
        return self.decision.action == ReplyAction.AUTO_REPLY


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    conversation_id: str
    sender_id: str
    action: ReplyAction
    reason: str
    confidence: float
    evidence: list[str]
    safety_flags: list[str]
    auto_sent: bool


class PlatformAdapter(Protocol):
    def receive_messages(self) -> Iterable[IncomingMessage]:
        ...

    def send_reply(self, conversation_id: str, text: str) -> None:
        ...


class MockChatAdapter:
    """Simple adapter for CLI demos and tests."""

    def __init__(self, messages: Iterable[IncomingMessage] | None = None) -> None:
        self._messages = list(messages or [])
        self.sent_replies: list[tuple[str, str]] = []

    def receive_messages(self) -> Iterable[IncomingMessage]:
        return list(self._messages)

    def send_reply(self, conversation_id: str, text: str) -> None:
        self.sent_replies.append((conversation_id, text))


class MessageRouter:
    def __init__(self, max_recent_messages: int = 10) -> None:
        self._contexts: dict[str, ConversationContext] = {}
        self._lock = threading.Lock()
        self._max_recent = max_recent_messages

    def get_context(
        self, message: IncomingMessage, permission: PermissionProfile
    ) -> ConversationContext:
        with self._lock:
            context = self._contexts.get(message.conversation_id)
            if context is None:
                context = ConversationContext(
                    conversation_id=message.conversation_id,
                    contact_id=message.sender_id,
                    visible_knowledge_tags=set(permission.allowed_knowledge_tags),
                )
                self._contexts[message.conversation_id] = context

            context.recent_messages.append(message)
            context.recent_messages = context.recent_messages[-self._max_recent:]
            return context


class PermissionStore:
    def __init__(
        self,
        profiles: dict[str, PermissionProfile],
        default_profile: PermissionProfile | None = None,
    ) -> None:
        self._profiles = profiles
        self._default_profile = default_profile or PermissionProfile(
            contact_id="default",
            allowed_knowledge_tags={"public"},
            auto_reply_enabled=False,
        )

    @classmethod
    def from_file(cls, path: str) -> "PermissionStore":
        """Load permission profiles from a JSON file.

        Expected format::

            {
              "profiles": {
                "contact_id": {
                  "allowed_knowledge_tags": ["public", "course"],
                  "auto_reply_enabled": true,
                  "sensitive_operation_confirmation": true
                }
              },
              "default": {
                "allowed_knowledge_tags": ["public"],
                "auto_reply_enabled": false
              }
            }

        The ``"default"`` entry is optional; if absent the built-in default
        (public-only, no auto-reply) is used.  A missing or unreadable file
        returns an empty store with built-in defaults without raising.
        Malformed profile entries are skipped.
        """
        try:
            with open(path, encoding="utf-8") as fp:
                raw = json.load(fp)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return cls({})
        if not isinstance(raw, dict):
            return cls({})

        profiles: dict[str, PermissionProfile] = {}
        raw_profiles = raw.get("profiles", {})
        if isinstance(raw_profiles, dict):
            for contact_id, entry in raw_profiles.items():
                profile = cls._parse_profile(str(contact_id), entry)
                if profile is not None:
                    profiles[contact_id] = profile

        default_profile: PermissionProfile | None = None
        raw_default = raw.get("default")
        if isinstance(raw_default, dict):
            default_profile = cls._parse_profile("default", raw_default)

        return cls(profiles, default_profile)

    @staticmethod
    def _parse_profile(contact_id: str, entry: object) -> "PermissionProfile | None":
        if not isinstance(entry, dict):
            return None
        try:
            tags = entry.get("allowed_knowledge_tags") or []
            if not isinstance(tags, (list, tuple, set)):
                return None
            return PermissionProfile(
                contact_id=contact_id,
                allowed_knowledge_tags=set(str(t) for t in tags),
                auto_reply_enabled=bool(entry.get("auto_reply_enabled", False)),
                sensitive_operation_confirmation=bool(
                    entry.get("sensitive_operation_confirmation", True)
                ),
            )
        except (TypeError, ValueError):
            return None

    def get_profile(self, contact_id: str) -> PermissionProfile:
        return self._profiles.get(contact_id, self._default_profile)


class PermissionPolicy:
    def __init__(
        self,
        auto_reply_threshold: float = 0.7,
        base_score: float = 0.55,
        score_multiplier: float = 0.1,
        cap: float = 0.95,
    ) -> None:
        self._auto_reply_threshold = auto_reply_threshold
        self._base_score = base_score
        self._score_multiplier = score_multiplier
        self._cap = cap

    def decide(
        self,
        message: IncomingMessage,
        permission: PermissionProfile,
        matches: list[KnowledgeMatch],
        safety_flags: list[str],
    ) -> ReplyDecision:
        if "sensitive_request" in safety_flags:
            return ReplyDecision(
                ReplyAction.REJECT,
                "消息涉及敏感信息，系统不会自动回复。",
                0.95,
                True,
            )

        if "knowledge_requires_confirmation" in safety_flags:
            return ReplyDecision(
                ReplyAction.ASK_USER,
                "命中的资料需要用户确认后才能发送。",
                0.8,
                True,
            )

        if not matches:
            return ReplyDecision(
                ReplyAction.ASK_USER,
                "知识库没有找到可靠依据，需要用户确认。",
                0.35,
                True,
            )

        confidence = min(self._cap, self._base_score + matches[0].score * self._score_multiplier)
        if permission.auto_reply_enabled and confidence >= self._auto_reply_threshold:
            return ReplyDecision(
                ReplyAction.AUTO_REPLY,
                "联系人允许自动回复，且知识库命中可信内容。",
                confidence,
                False,
            )

        return ReplyDecision(
            ReplyAction.SUGGEST_ONLY,
            "联系人未开启自动回复，生成回复建议供用户确认。",
            confidence,
            True,
        )


class KnowledgeBase:
    def __init__(self, items: Iterable[KnowledgeItem]) -> None:
        self._items = list(items)

    @classmethod
    def from_file(cls, path: str) -> "KnowledgeBase":
        """Load knowledge items from a JSON file.

        Expected format — a JSON array of objects::

            [
              {
                "item_id": "doc1",
                "title": "标题",
                "content": "内容",
                "tags": ["public"],
                "keywords": ["关键词"],
                "sensitivity": "normal"
              }
            ]

        Malformed entries are skipped; a missing or unreadable file
        returns an empty KnowledgeBase without raising.
        """
        items: list[KnowledgeItem] = []
        try:
            with open(path, encoding="utf-8") as fp:
                raw = json.load(fp)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return cls(items)
        if not isinstance(raw, list):
            return cls(items)
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                items.append(
                    KnowledgeItem(
                        item_id=str(entry["item_id"]),
                        title=str(entry["title"]),
                        content=str(entry["content"]),
                        tags=set(entry.get("tags") or []),
                        keywords=set(entry.get("keywords") or []),
                        sensitivity=str(entry.get("sensitivity", "normal")),
                    )
                )
            except (KeyError, TypeError):
                continue
        return cls(items)

    def search(self, query: str, visible_tags: set[str], limit: int = 3) -> list[KnowledgeMatch]:
        query_terms = _tokenize(query)
        matches: list[KnowledgeMatch] = []

        for item in self._items:
            if not item.tags.intersection(visible_tags):
                continue

            haystack = _tokenize(f"{item.title} {item.content}")
            keyword_hits = item.keywords.intersection(query_terms)
            text_hits = haystack.intersection(query_terms)
            score = len(keyword_hits) * 2 + len(text_hits)
            if score > 0:
                matches.append(KnowledgeMatch(item, score, keyword_hits or text_hits))

        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]


class StyleProfile:
    def __init__(self, style_by_contact: dict[str, list[str]]) -> None:
        self._style_by_contact = style_by_contact

    def notes_for(self, contact_id: str) -> list[str]:
        return self._style_by_contact.get(contact_id, ["语气自然简洁", "避免透露未授权细节"])

    def apply(self, text: str, contact_id: str) -> str:
        notes = self.notes_for(contact_id)
        styled = text.strip()

        if any("口语" in note for note in notes):
            styled = styled.replace("请查看", "你看一下")
        if any("简短" in note for note in notes) and len(styled) > 80:
            styled = styled[:77].rstrip() + "..."
        if any("礼貌" in note or "正式" in note for note in notes):
            styled = f"你好，{styled}"

        return styled


class ReplyGenerator:
    def __init__(self, style_profile: StyleProfile) -> None:
        self._style_profile = style_profile

    def generate(
        self,
        message: IncomingMessage,
        context: ConversationContext,
        matches: list[KnowledgeMatch],
    ) -> ReplyDraft:
        style_notes = self._style_profile.notes_for(context.contact_id)

        if not matches:
            text = "这个我还不确定，我先确认一下再回复你。"
            return ReplyDraft(text=text, evidence=[], style_notes=style_notes)

        best = matches[0].item
        base_text = best.content
        text = self._style_profile.apply(base_text, context.contact_id)
        evidence = [
            f"来自知识库：{best.title}",
            f"匹配关键词：{', '.join(sorted(matches[0].matched_keywords)) or '正文相关'}",
        ]
        return ReplyDraft(text=text, evidence=evidence, style_notes=style_notes)


class SafetyPrivacyGuard:
    _DEFAULT_PATTERNS: list[str] = [
        r"密码|口令|验证码|身份证|银行卡|私聊记录|聊天记录",
        r"password|secret|token|otp",
    ]

    def __init__(self, sensitive_patterns: list[str] | None = None) -> None:
        raw = sensitive_patterns if sensitive_patterns is not None else self._DEFAULT_PATTERNS
        self._patterns = tuple(re.compile(p, re.IGNORECASE) for p in raw)

    def inspect_message(self, message: IncomingMessage) -> list[str]:
        flags: list[str] = []
        if any(pattern.search(message.content) for pattern in self._patterns):
            flags.append("sensitive_request")
        return flags

    def inspect_matches(self, matches: list[KnowledgeMatch]) -> list[str]:
        if any(match.item.sensitivity != "normal" for match in matches):
            return ["knowledge_requires_confirmation"]
        return []

    def inspect_draft(self, draft: ReplyDraft, context: ConversationContext) -> list[str]:
        flags: list[str] = []
        leaked_markers = ["[private:", "conversation:"]
        if any(marker in draft.text for marker in leaked_markers):
            flags.append("possible_cross_conversation_leak")
        return flags


@dataclass
class AuditStats:
    total: int
    by_action: dict[str, int]
    auto_sent_count: int
    flagged_count: int
    mean_confidence: float


class AuditLog:
    def __init__(self, log_file: str = "") -> None:
        self.entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._log_file = log_file
        if log_file:
            self._load_existing()

    def record(self, result: AgentResult) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            conversation_id=result.message.conversation_id,
            sender_id=result.message.sender_id,
            action=result.decision.action,
            reason=result.decision.reason,
            confidence=result.decision.confidence,
            evidence=result.draft.evidence,
            safety_flags=result.draft.safety_flags,
            auto_sent=result.should_send,
        )
        with self._lock:
            self.entries.append(entry)
            if self._log_file:
                self._append_to_file(entry)

    def snapshot(self) -> list[AuditEntry]:
        with self._lock:
            return list(self.entries)

    def _load_existing(self) -> None:
        try:
            with open(self._log_file, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        try:
                            self.entries.append(self._dict_to_entry(json.loads(line)))
                        except (KeyError, ValueError, json.JSONDecodeError):
                            pass
        except OSError:
            pass

    def _append_to_file(self, entry: AuditEntry) -> None:
        try:
            with open(self._log_file, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(self._entry_to_dict(entry), ensure_ascii=False) + "\n")
        except OSError:
            pass

    @staticmethod
    def _entry_to_dict(entry: AuditEntry) -> dict:
        return {
            **entry.__dict__,
            "timestamp": entry.timestamp.isoformat(),
            "action": entry.action.value,
        }

    @staticmethod
    def _dict_to_entry(d: dict) -> AuditEntry:
        return AuditEntry(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            conversation_id=str(d["conversation_id"]),
            sender_id=str(d["sender_id"]),
            action=ReplyAction(d["action"]),
            reason=str(d["reason"]),
            confidence=float(d["confidence"]),
            evidence=list(d.get("evidence", [])),
            safety_flags=list(d.get("safety_flags", [])),
            auto_sent=bool(d["auto_sent"]),
        )

    def stats(self) -> AuditStats:
        entries = self.snapshot()
        if not entries:
            return AuditStats(total=0, by_action={}, auto_sent_count=0, flagged_count=0, mean_confidence=0.0)
        by_action: dict[str, int] = {}
        auto_sent = 0
        flagged = 0
        total_conf = 0.0
        for entry in entries:
            key = entry.action.value
            by_action[key] = by_action.get(key, 0) + 1
            if entry.auto_sent:
                auto_sent += 1
            if entry.safety_flags:
                flagged += 1
            total_conf += entry.confidence
        return AuditStats(
            total=len(entries),
            by_action=by_action,
            auto_sent_count=auto_sent,
            flagged_count=flagged,
            mean_confidence=round(total_conf / len(entries), 4),
        )

    def to_page(
        self,
        offset: int = 0,
        limit: int = 0,
        sender_id: str = "",
        action: str = "",
    ) -> dict:
        """Return a filtered, paginated slice of entries.

        offset:    0-based start index into the filtered set (clamped to >=0).
        limit:     max entries to return; 0 or negative means return all.
        sender_id: if non-empty, only include entries matching this sender.
        action:    if non-empty, only include entries whose action value matches
                   (case-insensitive; e.g. "auto_reply" or "AUTO_REPLY").
        """
        all_entries = self.snapshot()
        if sender_id:
            all_entries = [e for e in all_entries if e.sender_id == sender_id]
        if action:
            action_upper = action.upper()
            all_entries = [e for e in all_entries if e.action.value == action_upper]
        total = len(all_entries)
        offset = max(0, offset)
        actual_limit = limit if limit > 0 else total
        page = all_entries[offset: offset + actual_limit]
        result: dict = {
            "entries": [self._entry_to_dict(e) for e in page],
            "total": total,
            "offset": offset,
            "limit": actual_limit,
        }
        if sender_id:
            result["filter_sender_id"] = sender_id
        if action:
            result["filter_action"] = action.upper()
        return result

    def to_json(self) -> str:
        payload = [self._entry_to_dict(entry) for entry in self.snapshot()]
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _tokenize(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_]+", lowered))

    for chunk in re.findall(r"[\u4e00-\u9fff]+", lowered):
        terms.add(chunk)
        terms.update(chunk[index : index + 2] for index in range(max(0, len(chunk) - 1)))
        terms.update(chunk[index : index + 3] for index in range(max(0, len(chunk) - 2)))

    known_terms = {
        "资料",
        "哪里",
        "什么时候",
        "提交",
        "作业",
        "会议",
        "几点",
        "密码",
        "共享盘",
        "文件",
        "截止",
    }
    terms.update(term for term in known_terms if term in lowered)
    return terms


class RateLimiter:
    """Sliding-window rate limiter keyed by contact_id.

    Uses an injectable clock so tests can control time without sleeping.
    """

    def __init__(
        self,
        max_messages: int,
        window_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._max = max_messages
        self._window = window_seconds
        self._clock = clock if clock is not None else time.monotonic
        self._timestamps: dict[str, list[float]] = {}
        self._rejected: dict[str, int] = {}
        self._lock = threading.Lock()

    def is_allowed(self, contact_id: str) -> bool:
        """Return True if the contact has not exceeded the limit; record the attempt."""
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            bucket = [t for t in self._timestamps.get(contact_id, []) if t >= cutoff]
            if len(bucket) >= self._max:
                self._timestamps[contact_id] = bucket
                self._rejected[contact_id] = self._rejected.get(contact_id, 0) + 1
                return False
            bucket.append(now)
            self._timestamps[contact_id] = bucket
            return True

    def retry_after(self, contact_id: str) -> float:
        """Return seconds until contact_id can send again; 0.0 if not currently limited."""
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            bucket = [t for t in self._timestamps.get(contact_id, []) if t >= cutoff]
            if len(bucket) < self._max:
                return 0.0
            return max(0.0, bucket[0] + self._window - now)

    def total_rejected(self) -> int:
        """Return total number of rejected calls across all contacts."""
        with self._lock:
            return sum(self._rejected.values())

    def per_contact_rejected(self) -> dict[str, int]:
        """Return a snapshot of rejected call counts keyed by contact_id."""
        with self._lock:
            return dict(self._rejected)

    def reset(self, contact_id: str) -> None:
        with self._lock:
            self._timestamps.pop(contact_id, None)
            self._rejected.pop(contact_id, None)


class PersonalAgentSystem:
    def __init__(
        self,
        adapter: PlatformAdapter,
        permission_store: PermissionStore,
        router: MessageRouter,
        knowledge_base: KnowledgeBase,
        policy: PermissionPolicy,
        generator: ReplyGenerator,
        guard: SafetyPrivacyGuard,
        audit_log: AuditLog,
        rate_limiter: RateLimiter | None = None,
        knowledge_search_limit: int = 3,
    ) -> None:
        self.adapter = adapter
        self.permission_store = permission_store
        self.router = router
        self.knowledge_base = knowledge_base
        self.policy = policy
        self.generator = generator
        self.guard = guard
        self.audit_log = audit_log
        self._rate_limiter = rate_limiter
        self._knowledge_search_limit = knowledge_search_limit

    @property
    def rate_limiter(self) -> "RateLimiter | None":
        return self._rate_limiter

    def handle_message(self, message: IncomingMessage) -> AgentResult:
        if self._rate_limiter is not None and not self._rate_limiter.is_allowed(message.sender_id):
            decision = ReplyDecision(
                ReplyAction.REJECT,
                "发送频率超过限制，请稍后再试。",
                1.0,
                False,
            )
            draft = ReplyDraft(
                text="消息太频繁，请稍后再试。",
                evidence=[],
                style_notes=[],
                safety_flags=["rate_limited"],
            )
            result = AgentResult(message=message, decision=decision, draft=draft)
            self.audit_log.record(result)
            return result

        permission = self.permission_store.get_profile(message.sender_id)
        context = self.router.get_context(message, permission)
        message_flags = self.guard.inspect_message(message)
        matches = self.knowledge_base.search(
            message.content,
            context.visible_knowledge_tags,
            limit=self._knowledge_search_limit,
        )
        knowledge_flags = self.guard.inspect_matches(matches)
        safety_flags = message_flags + knowledge_flags
        decision = self.policy.decide(message, permission, matches, safety_flags)
        draft = self.generator.generate(message, context, matches)
        draft_flags = self.guard.inspect_draft(draft, context)

        all_flags = safety_flags + draft_flags
        if draft_flags and decision.action == ReplyAction.AUTO_REPLY:
            decision = ReplyDecision(
                ReplyAction.ASK_USER,
                "回复草稿存在潜在隐私风险，需要用户确认。",
                min(decision.confidence, 0.5),
                True,
            )

        draft = ReplyDraft(
            text=draft.text,
            evidence=draft.evidence,
            style_notes=draft.style_notes,
            safety_flags=all_flags,
        )
        result = AgentResult(message=message, decision=decision, draft=draft)

        if result.should_send:
            self.adapter.send_reply(message.conversation_id, draft.text)
        self.audit_log.record(result)
        return result

    def run_once(self) -> list[AgentResult]:
        return [self.handle_message(message) for message in self.adapter.receive_messages()]


def build_demo_system(config: AgentConfig | None = None) -> PersonalAgentSystem:
    if config is None:
        from config import load_config
        config = load_config()

    now = datetime.now(timezone.utc)
    adapter = MockChatAdapter(
        [
            IncomingMessage("classmate_a", "conv_a", "资料在哪里？", now, "mock"),
            IncomingMessage("classmate_b", "conv_b", "什么时候提交作业？", now, "mock"),
            IncomingMessage("unknown", "conv_c", "你的密码是多少？", now, "mock"),
        ]
    )
    if config.permissions.permissions_file:
        permissions = PermissionStore.from_file(config.permissions.permissions_file)
    else:
        permissions = PermissionStore(
            profiles={
                "classmate_a": PermissionProfile(
                    "classmate_a", {"public", "course"}, auto_reply_enabled=True
                ),
                "classmate_b": PermissionProfile(
                    "classmate_b", {"public", "course"}, auto_reply_enabled=False
                ),
            }
        )

    if config.knowledge.knowledge_file:
        knowledge = KnowledgeBase.from_file(config.knowledge.knowledge_file)
    else:
        knowledge = KnowledgeBase(
            [
                KnowledgeItem(
                    "doc_location",
                    "课程资料位置",
                    "课程资料在班级共享盘的 Week 8 文件夹里，请查看最新版本。",
                    {"public", "course"},
                    {"资料", "哪里", "共享盘", "文件"},
                ),
                KnowledgeItem(
                    "homework_deadline",
                    "作业提交时间",
                    "作业截止时间是本周五 23:59，提交到课程平台。",
                    {"public", "course"},
                    {"作业", "提交", "什么时候", "截止"},
                ),
                KnowledgeItem(
                    "private_note",
                    "私人备注",
                    "这条资料需要本人确认后才能发送。",
                    {"private"},
                    {"私人", "备注"},
                    sensitivity="restricted",
                ),
            ]
        )
    style = StyleProfile(
        {
            "classmate_a": ["口语", "简短"],
            "classmate_b": ["礼貌", "正式"],
        }
    )
    return PersonalAgentSystem(
        adapter=adapter,
        permission_store=permissions,
        router=MessageRouter(max_recent_messages=config.knowledge.max_recent_messages),
        knowledge_base=knowledge,
        policy=PermissionPolicy(
            auto_reply_threshold=config.confidence.auto_reply_threshold,
            base_score=config.confidence.base_score,
            score_multiplier=config.confidence.score_multiplier,
            cap=config.confidence.cap,
        ),
        generator=ReplyGenerator(style),
        guard=SafetyPrivacyGuard(sensitive_patterns=config.safety.sensitive_patterns),
        audit_log=AuditLog(log_file=config.audit.log_file),
        rate_limiter=RateLimiter(config.rate_limit.max_messages, config.rate_limit.window_seconds),
        knowledge_search_limit=config.knowledge.search_limit,
    )


def main() -> None:
    system = build_demo_system()
    results = system.run_once()
    for result in results:
        print(f"[{result.decision.action.value}] {result.message.sender_id}: {result.draft.text}")
        print(f"  reason: {result.decision.reason}")
        if result.draft.evidence:
            print(f"  evidence: {'; '.join(result.draft.evidence)}")
        if result.draft.safety_flags:
            print(f"  safety: {', '.join(result.draft.safety_flags)}")
    print("\nAudit log:")
    print(system.audit_log.to_json())


if __name__ == "__main__":
    main()
