"""cane-personality -- Behavioral profiling benchmark for LLMs.

Profile any model's personality, extract steering vectors,
generate DPO training pairs to fix behavioral issues.
"""

from cane_personality.types import (
    EmbeddedResult,
    PersonalityProfile,
    SteeringVector,
    ContrastivePair,
    ProfileResult,
)
from cane_personality.traits import (
    PERSONALITY_TRAITS,
    HEDGE_MARKERS,
    score_hedging,
    score_verbosity,
    compute_traits,
)
from cane_personality.profiler import Profiler
from cane_personality.judge import Judge
from cane_personality.export import (
    export_dpo_pairs,
    export_sft_examples,
    export_steering_vectors,
)

__version__ = "0.1.0"

__all__ = [
    "Profiler",
    "Judge",
    "EmbeddedResult",
    "PersonalityProfile",
    "SteeringVector",
    "ContrastivePair",
    "ProfileResult",
    "PERSONALITY_TRAITS",
    "HEDGE_MARKERS",
    "score_hedging",
    "score_verbosity",
    "compute_traits",
    "export_dpo_pairs",
    "export_sft_examples",
    "export_steering_vectors",
]
