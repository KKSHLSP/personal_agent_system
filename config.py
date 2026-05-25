"""Configuration for the personal agent system.

Load config from a JSON or YAML file (partial merge — only specified keys
are overridden), then apply environment variable overrides which always win.

JSON example (agent.json):
    {
      "rate_limit":  {"max_messages": 10, "window_seconds": 30},
      "confidence":  {"auto_reply_threshold": 0.8},
      "audit":       {"log_file": "/var/log/agent_audit.jsonl", "max_entries": 10000},
      "permissions": {"permissions_file": "/etc/agent/permissions.json"},
      "webui":       {"port": 9090}
    }

YAML example (agent.yaml) — requires: pip install pyyaml:
    rate_limit:
      max_messages: 10
      window_seconds: 30
    audit:
      log_file: /var/log/agent_audit.jsonl
      max_entries: 10000
    permissions:
      permissions_file: /etc/agent/permissions.json
    webui:
      port: 9090

Environment variable overrides (all optional):
    AGENT_RATE_LIMIT_MAX_MESSAGES         int
    AGENT_RATE_LIMIT_WINDOW_SECONDS       float
    AGENT_CONFIDENCE_AUTO_REPLY_THRESHOLD float
    AGENT_CONFIDENCE_BASE_SCORE           float
    AGENT_CONFIDENCE_SCORE_MULTIPLIER     float
    AGENT_CONFIDENCE_CAP                  float
    AGENT_KNOWLEDGE_SEARCH_LIMIT          int
    AGENT_KNOWLEDGE_MAX_RECENT_MESSAGES   int
    AGENT_KNOWLEDGE_FILE                  str   (empty = use built-in demo items)
    AGENT_AUDIT_LOG_FILE                  str   (empty = disabled)
    AGENT_AUDIT_MAX_ENTRIES               int   (0 = unlimited; evicts oldest when exceeded)
    AGENT_PERMISSIONS_FILE                str   (empty = use built-in demo profiles)
    AGENT_WEBUI_HOST                      str
    AGENT_WEBUI_PORT                      int
    AGENT_WEBUI_LOCAL_AI_URL              str
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    max_messages: int = 5
    window_seconds: float = 60.0


@dataclass
class ConfidenceConfig:
    auto_reply_threshold: float = 0.7
    base_score: float = 0.55
    score_multiplier: float = 0.1
    cap: float = 0.95


@dataclass
class SafetyConfig:
    sensitive_patterns: list[str] = field(default_factory=lambda: [
        r"密码|口令|验证码|身份证|银行卡|私聊记录|聊天记录",
        r"password|secret|token|otp",
    ])


@dataclass
class KnowledgeConfig:
    search_limit: int = 3
    max_recent_messages: int = 10
    knowledge_file: str = ""


@dataclass
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    local_ai_url: str = "http://127.0.0.1:8001"


@dataclass
class AuditConfig:
    log_file: str = ""
    max_entries: int = 0


@dataclass
class PermissionsConfig:
    permissions_file: str = ""


@dataclass
class AgentConfig:
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)

    def validate(self) -> None:
        """Raise ValueError if any field is outside its valid range."""
        if self.rate_limit.max_messages < 1:
            raise ValueError("rate_limit.max_messages must be >= 1")
        if self.rate_limit.window_seconds <= 0:
            raise ValueError("rate_limit.window_seconds must be > 0")
        if not 0.0 <= self.confidence.auto_reply_threshold <= 1.0:
            raise ValueError("confidence.auto_reply_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.confidence.base_score <= 1.0:
            raise ValueError("confidence.base_score must be in [0.0, 1.0]")
        if self.confidence.score_multiplier < 0:
            raise ValueError("confidence.score_multiplier must be >= 0")
        if not 0.0 <= self.confidence.cap <= 1.0:
            raise ValueError("confidence.cap must be in [0.0, 1.0]")
        if self.knowledge.search_limit < 1:
            raise ValueError("knowledge.search_limit must be >= 1")
        if self.knowledge.max_recent_messages < 1:
            raise ValueError("knowledge.max_recent_messages must be >= 1")
        if self.audit.max_entries < 0:
            raise ValueError("audit.max_entries must be >= 0")
        if not 1 <= self.webui.port <= 65535:
            raise ValueError("webui.port must be in [1, 65535]")


def config_to_dict(config: AgentConfig) -> dict:
    """Serialize an AgentConfig to a redacted, JSON-safe dict for the /api/config endpoint.

    File paths are replaced by a boolean ``file_enabled`` flag so internal
    filesystem layout is never leaked.  ``sensitive_patterns`` is replaced by
    ``pattern_count`` so detection rules cannot be reverse-engineered via the
    API.
    """
    return {
        "rate_limit": {
            "max_messages": config.rate_limit.max_messages,
            "window_seconds": config.rate_limit.window_seconds,
        },
        "confidence": {
            "auto_reply_threshold": config.confidence.auto_reply_threshold,
            "base_score": config.confidence.base_score,
            "score_multiplier": config.confidence.score_multiplier,
            "cap": config.confidence.cap,
        },
        "safety": {
            "pattern_count": len(config.safety.sensitive_patterns),
        },
        "knowledge": {
            "search_limit": config.knowledge.search_limit,
            "max_recent_messages": config.knowledge.max_recent_messages,
            "file_enabled": bool(config.knowledge.knowledge_file),
        },
        "audit": {
            "file_enabled": bool(config.audit.log_file),
            "max_entries": config.audit.max_entries,
        },
        "permissions": {
            "file_enabled": bool(config.permissions.permissions_file),
        },
        "webui": {
            "host": config.webui.host,
            "port": config.webui.port,
            "local_ai_url": _redact_url_userinfo(config.webui.local_ai_url),
        },
    }


def _redact_url_userinfo(value: str) -> str:
    parts = urlsplit(value)
    if not parts.username and not parts.password:
        return value
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def load_config(path: str | None = None) -> AgentConfig:
    """Return an AgentConfig from *path* (JSON/YAML) plus env-var overrides.

    A missing or unreadable file silently returns all defaults.
    Raises ValueError if the resulting config contains out-of-range values.
    """
    config = AgentConfig()
    if path is not None:
        raw = _load_file(path)
        if raw:
            _apply_raw(config, raw)
    _apply_env(config)
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            content = fp.read()
    except OSError as exc:
        logger.warning("Cannot read config file %r: %s", path, exc)
        return {}
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import]
            return yaml.safe_load(content) or {}
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for .yaml/.yml config files: pip install pyyaml"
            ) from exc
    return json.loads(content or "{}") or {}


def _apply_raw(config: AgentConfig, raw: dict) -> None:
    section_map: dict[str, object] = {
        "rate_limit": config.rate_limit,
        "confidence": config.confidence,
        "safety": config.safety,
        "knowledge": config.knowledge,
        "audit": config.audit,
        "permissions": config.permissions,
        "webui": config.webui,
    }
    for section_name, section_raw in raw.items():
        if section_name in section_map and isinstance(section_raw, dict):
            _merge_section(section_map[section_name], section_raw)


def _merge_section(obj: object, raw: dict) -> None:
    for key, value in raw.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _apply_env(config: AgentConfig) -> None:
    _env_int("AGENT_RATE_LIMIT_MAX_MESSAGES", config.rate_limit, "max_messages")
    _env_float("AGENT_RATE_LIMIT_WINDOW_SECONDS", config.rate_limit, "window_seconds")
    _env_float("AGENT_CONFIDENCE_AUTO_REPLY_THRESHOLD", config.confidence, "auto_reply_threshold")
    _env_float("AGENT_CONFIDENCE_BASE_SCORE", config.confidence, "base_score")
    _env_float("AGENT_CONFIDENCE_SCORE_MULTIPLIER", config.confidence, "score_multiplier")
    _env_float("AGENT_CONFIDENCE_CAP", config.confidence, "cap")
    _env_int("AGENT_KNOWLEDGE_SEARCH_LIMIT", config.knowledge, "search_limit")
    _env_int("AGENT_KNOWLEDGE_MAX_RECENT_MESSAGES", config.knowledge, "max_recent_messages")
    _env_str("AGENT_KNOWLEDGE_FILE", config.knowledge, "knowledge_file")
    _env_str("AGENT_AUDIT_LOG_FILE", config.audit, "log_file")
    _env_int("AGENT_AUDIT_MAX_ENTRIES", config.audit, "max_entries")
    _env_str("AGENT_PERMISSIONS_FILE", config.permissions, "permissions_file")
    _env_str("AGENT_WEBUI_HOST", config.webui, "host")
    _env_int("AGENT_WEBUI_PORT", config.webui, "port")
    _env_str("AGENT_WEBUI_LOCAL_AI_URL", config.webui, "local_ai_url")


def _env_int(key: str, obj: object, attr: str) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setattr(obj, attr, int(val))
        except ValueError:
            pass


def _env_float(key: str, obj: object, attr: str) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setattr(obj, attr, float(val))
        except ValueError:
            pass


def _env_str(key: str, obj: object, attr: str) -> None:
    val = os.environ.get(key)
    if val is not None:
        setattr(obj, attr, val)
