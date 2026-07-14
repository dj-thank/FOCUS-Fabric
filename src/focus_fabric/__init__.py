"""FOCUS-Fabric public research implementation."""
from .agent_memory import CompactionSnapshot, MemoryKind, MemoryRecord, TrajectoryLedger
from .certificate import ConformalCertificate
from .config import CompilerConfig, FabricConfig
from .exact import exact_head_batch, exact_head_summary, exact_multihead_summary
from .fabric import MemoryFabricLayer
from .integration import FabricGenerationResult, FabricModelCache, generate_fabric, sequential_logits_fabric
from .monitoring import DriftSentinel
from .verified import VerifiedDecodeResult, greedy_decode, verified_block_decode
from .trace_capture import SDPATrace, capture_sdpa_traces
from .training import NativeLossResult, NativeLossWeights, focus_native_loss
from .types import AttentionSummary, CodecMetrics, FabricStats, merge_summaries

__all__ = [
    "AttentionSummary",
    "CodecMetrics",
    "CompilerConfig",
    "CompactionSnapshot",
    "ConformalCertificate",
    "FabricConfig",
    "FabricStats",
    "DriftSentinel",
    "FabricGenerationResult",
    "FabricModelCache",
    "MemoryFabricLayer",
    "SDPATrace",
    "VerifiedDecodeResult",
    "generate_fabric",
    "greedy_decode",
    "sequential_logits_fabric",
    "verified_block_decode",
    "MemoryKind",
    "MemoryRecord",
    "NativeLossResult",
    "NativeLossWeights",
    "TrajectoryLedger",
    "capture_sdpa_traces",
    "exact_head_batch",
    "exact_head_summary",
    "exact_multihead_summary",
    "focus_native_loss",
    "merge_summaries",
]

__version__ = "0.2.0"
