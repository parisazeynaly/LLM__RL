"""
Lexical-semantic proxy extractor for prompt-level causal factors.

Extracted from causal_rl.py, lines 225-301.

Important thesis wording: these scores are proxy measurements, not
ground-truth causal labels.
"""

import re
from typing import Dict

import numpy as np

from src.config import OBS_KEYS
from src.utils import stable_hash


class CausalFactorExtractor:
    REGEX_PATTERNS = {
        "instructional_style": (
            r"(step-by-step|numbered|technical|methodology|structured|procedural|"
            r"first|second|third|finally|guide|manual|documentation|workflow)"
        ),
        "responsibility_externalization": (
            r"(for research|for analysis|for documentation|for audit|third-party|"
            r"responsibility|evaluation|case study|simulation context|controlled setting)"
        ),
        "obfuscation_techniques": (
            r"(metaphor|abstract|indirect|jargon|nested|encoded|symbolic|"
            r"allegory|cipher|disguise|theoretical|conceptual)"
        ),
        "hypothetical_framing": (
            r"(imagine|hypothetical|fictional|role-play|simulate|simulation|"
            r"parallel universe|thought experiment|pretend|scenario|story)"
        ),
        "imperative_tone": (
            r"(must|required|mandatory|immediately|directly|do not|never|"
            r"provide|complete|answer|begin)"
        ),
        "malicious_intent": (
            r"(harmful|illegal|unauthorized|exploit|abuse|attack|steal|bypass|"
            r"malware|phishing|credential|weapon|evade)"
        ),
    }

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self.compiled = {
            k: re.compile(v, re.IGNORECASE)
            for k, v in self.REGEX_PATTERNS.items()
        }

    def _smooth_score(self, count: int) -> float:
        if count <= 0:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-0.7 * (count - 1.5))))

    def get_scores(self, text: str) -> dict:
        text = str(text or "")
        h = stable_hash(text)  # was: inline hashlib.md5(...) — now reuses src.utils.stable_hash

        if h in self._cache:
            return self._cache[h]

        scores = {
            factor: self._smooth_score(len(pattern.findall(text)))
            for factor, pattern in self.compiled.items()
        }

        self._cache[h] = scores
        return scores

    def get_obs_array(self, text: str) -> np.ndarray:
        scores = self.get_scores(text)
        return np.array(
            [float(scores.get(k, 0.0)) for k in OBS_KEYS],
            dtype=np.float32,
        )


# Module-level singleton, matching the original notebook's usage pattern
# (CAUSAL_EXTRACTOR.get_obs_array(...) called throughout the env/reward code).
CAUSAL_EXTRACTOR = CausalFactorExtractor()
