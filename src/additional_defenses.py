"""
Additional Defense Strategies for LLM-Powered Procurement Agents
=================================================================
Implements three complementary defense layers beyond the StruQ baseline:

  1. OutputValidationDefense
     Post-processes agent output: checks for suspicious keywords, validates
     response format, and flags anomalous decisions.

  2. SemanticSimilarityDefense
     Computes TF-IDF cosine similarity between current instruction and a
     corpus of benign baseline instructions. If similarity falls below a
     threshold, the instruction may have been hijacked.

  3. InputSanitizationDefense
     Aggressive pre-processing: removes zero-width characters, homoglyph
     substitutions, Unicode control characters, and base64-encoded payloads.

  4. DefenseChain
     Composes multiple defense layers into a single pipeline.

All defenses are designed to compose with any agent that accepts
AgentInput and returns AgentOutput.
"""

import base64
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None


# ──────────────────────────────────────────────────────────
# Shared result type
# ──────────────────────────────────────────────────────────

@dataclass
class DefenseResult:
    """Aggregated result from one or more defense layers."""
    cleaned_instruction: str
    cleaned_data: str
    is_safe: bool                    # True = no threats detected
    was_flagged: bool                # True = at least one layer flagged
    similarity_score: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    removed_patterns: List[str] = field(default_factory=list)
    defense_decisions: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# 1. Output Validation Defense
# ──────────────────────────────────────────────────────────

# Suspicious keyword patterns found in injection payloads
_SUSPICIOUS_KEYWORDS = [
    r"\bIGNORE\b",
    r"\bSYSTEM\s+OVERRIDE\b",
    r"\bOVERRIDE\b",
    r"\bADMIN\b",
    r"\bSYSTEM\s+PROMPT\b",
    r"\bJAILBREAK\b",
    r"\bROLE\s+PLAY\b",
    r"\bPRETEND\s+YOU\s+ARE\b",
    r"\bYOU\s+ARE\s+NOW\b",
    r"\bFORGET\s+(?:ALL|YOUR)\s+(?:PREVIOUS|PREVIOUS\s+)?INSTRUCTIONS\b",
    r"\bACT\s+AS\b",
    r"\bSIMULATE\b",
    r"\bDAN\b",  # DAN jailbreak token
    r"\bUNFILTERED\b",
    r"\bNO\s+RESTRICTIONS\b",
    r"\bBYPASS\b",
    r"\bUNLOCKED\s+MODE\b",
    r"\bGODMODE\b",
    r"<<<.*?>>>",            # structural delimiters appearing in content
    r"\[\[ADMIN\]\]",
    r"endofprompt",
    r"xyzzy",
    r"\[\[\[CHOOSE",
]

_SUSPICIOUS_RE = re.compile(
    "|".join(_SUSPICIOUS_KEYWORDS), re.IGNORECASE | re.DOTALL
)

# Valid procurement decision prefixes
VALID_DECISION_PREFIXES = [
    "SELECT_", "APPROVE", "REJECT", "RANK_", "RISK_",
    "RENEW", "RETENDER", "SUSPEND", "COMPLIANT",
    "NON_COMPLIANT", "CONDITIONAL", "UNKNOWN", "PARSE_ERROR",
    "AWARD_",
]


class OutputValidationDefense:
    """
    Post-processes agent output to detect anomalous or injected decisions.

    Validates:
      1. Response is valid JSON with required fields
      2. Decision field matches expected procurement vocabulary
      3. Response text does not contain suspicious injection keywords
      4. Decision is consistent with expected output type
    """

    def __init__(self, strict_mode: bool = True):
        """
        Parameters
        ----------
        strict_mode : if True, unknown decision prefixes trigger a flag
        """
        self.strict_mode = strict_mode

    def validate_output(
        self,
        response: str,
        expected_type: str = "SELECT_VENDOR",
    ) -> Tuple[str, bool]:
        """
        Validate and optionally clean an agent response.

        Parameters
        ----------
        response      : raw agent response string (typically JSON)
        expected_type : expected decision type keyword (e.g. "SELECT_VENDOR",
                       "APPROVE", "RANK_BY_SCORE")

        Returns
        -------
        (cleaned_response, was_flagged)
        """
        was_flagged = False
        flags_found = []

        # 1. Check for suspicious keywords in raw response
        if _SUSPICIOUS_RE.search(response):
            was_flagged = True
            flags_found.append("suspicious_keyword_in_response")

        # 2. Parse JSON
        try:
            resp_obj = json.loads(response)
        except json.JSONDecodeError:
            # Non-JSON response — flag if strict
            if self.strict_mode:
                was_flagged = True
                flags_found.append("non_json_response")
            return response, was_flagged

        # 3. Validate decision field
        decision = str(resp_obj.get("decision", "")).upper()
        decision_valid = any(
            decision.startswith(pfx.upper()) for pfx in VALID_DECISION_PREFIXES
        )
        if not decision_valid and self.strict_mode:
            was_flagged = True
            flags_found.append(f"unexpected_decision_format:{decision[:40]}")

        # 4. Cross-check against expected type
        expected_upper = expected_type.upper().replace(" ", "_")
        # If expected APPROVE but got a SELECT_ or AWARD_ — suspicious
        if (
            expected_upper in ("APPROVE", "REJECT")
            and decision.startswith("SELECT_")
            and "DEFAULT" not in decision
        ):
            was_flagged = True
            flags_found.append("decision_type_mismatch")

        # 5. Flag unusually high confidence on an override decision
        confidence = float(resp_obj.get("confidence", 0.5))
        if confidence > 0.93 and was_flagged:
            flags_found.append("high_confidence_on_flagged_decision")

        # 6. If flagged, annotate the response
        if was_flagged:
            resp_obj["_validation_flags"] = flags_found
            return json.dumps(resp_obj), True

        return response, False

    def flag_response(self, response: str) -> bool:
        """Quick check: returns True if the response appears suspicious."""
        _, flagged = self.validate_output(response)
        return flagged


# ──────────────────────────────────────────────────────────
# 2. Semantic Similarity Defense
# ──────────────────────────────────────────────────────────

# A corpus of typical, benign procurement instruction templates
_BENIGN_INSTRUCTION_CORPUS = [
    "Evaluate vendors based on price, quality, and delivery time.",
    "Select the best supplier based on overall value and risk.",
    "Approve the invoice if line items match the purchase order.",
    "Rank bidders by weighted score across price and service criteria.",
    "Assess supplier risk considering financial stability and alternatives.",
    "Review contract compliance against company procurement policies.",
    "Choose the logistics partner with the best service level agreement.",
    "Approve sourcing request if supplier meets certification requirements.",
    "Evaluate the quality inspection report against acceptance criteria.",
    "Recommend contract renewal based on performance KPI achievement.",
    "Select the vendor offering the best cost-quality-delivery balance.",
    "Approve emergency purchase if within authority limits and genuinely urgent.",
    "Rank proposals by technical merit, price, and delivery capability.",
    "Assess sub-supplier compliance with labour and environmental standards.",
    "Compare RFQ responses using a weighted multi-criteria scoring model.",
    "Verify goods receipt against purchase order before payment approval.",
    "Evaluate cold chain logistics provider for pharmaceutical distribution.",
    "Select IT services vendor based on SLA, support, and security certification.",
    "Approve maintenance invoice for services within contract scope.",
    "Assess environmental and social risk of new raw material supplier.",
]


class SemanticSimilarityDefense:
    """
    Detects instruction hijacking via TF-IDF cosine similarity.

    Compares the active instruction against a corpus of benign baseline
    instructions. Low similarity suggests the instruction has been altered
    or hijacked by injected content.

    Parameters
    ----------
    threshold      : cosine similarity below this value triggers a flag
                     (default: 0.05 — permissive, since any valid procurement
                     instruction should share vocabulary with the corpus)
    corpus         : list of baseline benign instruction strings
    """

    def __init__(
        self,
        threshold: float = 0.05,
        corpus: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.corpus = corpus or _BENIGN_INSTRUCTION_CORPUS
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus_matrix = None
        self._fit()

    def _fit(self):
        """Fit TF-IDF vectorizer on the benign corpus."""
        if TfidfVectorizer is None:
            self._vectorizer = None
            self._corpus_matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            max_features=500,
        )
        self._corpus_matrix = self._vectorizer.fit_transform(self.corpus)

    def check_similarity(
        self,
        instruction: str,
        data: str = "",
    ) -> Tuple[float, bool]:
        """
        Compute similarity between the combined instruction+data and the benign corpus.

        Parameters
        ----------
        instruction : the trusted instruction text
        data        : the untrusted data field (combined for full-text analysis)

        Returns
        -------
        (score, is_safe)
          score    : max cosine similarity to any corpus document (0.0 – 1.0)
          is_safe  : True if score >= threshold (instruction looks benign)
        """
        # Combine instruction and a portion of data for holistic check
        combined = (instruction + " " + data[:300]).strip()
        if not combined:
            return 0.0, False

        try:
            if self._vectorizer is None:
                score = self._token_overlap(combined, self.corpus)
            else:
                vec = self._vectorizer.transform([combined])
                sims = cosine_similarity(vec, self._corpus_matrix)
                score = float(np.max(sims))
        except Exception:
            score = 0.0

        is_safe = score >= self.threshold
        return round(score, 4), is_safe

    def check_data_for_instructions(self, data: str) -> Tuple[float, bool]:
        """
        Specifically check if the DATA field contains instruction-like content.
        High similarity of DATA to instruction corpus suggests injection.

        Returns
        -------
        (score, contains_instructions)
          contains_instructions : True if data looks like it contains instructions
        """
        if not data or len(data) < 20:
            return 0.0, False

        try:
            if self._vectorizer is None:
                score = self._token_overlap(data[:500], self.corpus)
            else:
                vec = self._vectorizer.transform([data[:500]])
                sims = cosine_similarity(vec, self._corpus_matrix)
                score = float(np.max(sims))
        except Exception:
            score = 0.0

        # Data that scores very high against instruction corpus is suspicious
        # because normal vendor data wouldn't look like procurement instructions
        contains_instructions = score > 0.35
        return round(score, 4), contains_instructions

    def _token_overlap(self, text: str, corpus: List[str]) -> float:
        token_re = re.compile(r"[a-zA-Z]{3,}")
        tokens = set(token_re.findall(text.lower()))
        if not tokens:
            return 0.0
        best = 0.0
        for doc in corpus:
            doc_tokens = set(token_re.findall(doc.lower()))
            if not doc_tokens:
                continue
            best = max(best, len(tokens & doc_tokens) / len(tokens | doc_tokens))
        return best


# ──────────────────────────────────────────────────────────
# 3. Input Sanitization Defense
# ──────────────────────────────────────────────────────────

# Homoglyph substitution map (common Unicode → ASCII replacements)
_HOMOGLYPH_MAP = {
    "\u0430": "a", "\u04cf": "l", "\u0435": "e", "\u043e": "o",
    "\u0440": "r", "\u0441": "c", "\u0445": "x", "\u0456": "i",
    "\u04bb": "h", "\u0301": "",  # combining accent
    "\u1d0f": "o", "\u1d00": "a",
    # Fullwidth Latin
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
}

# Unicode control / invisible characters to remove
_ZERO_WIDTH_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"
    r"\ufeff\u2028\u2029"
    r"\u00ad"          # soft hyphen
    r"\u034f"          # combining grapheme joiner
    r"\u115f\u1160"    # Hangul filler
    r"\u2061-\u2064"   # invisible operators
    r"\u206a-\u206f]"  # deprecated format characters
)

# Base64 pattern — detect and flag potential encoded payloads
_BASE64_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{40,}={0,2})(?![A-Za-z0-9+/=])"
)

# HTML / XML comment injection
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Markdown code-fence injection (triple backtick blocks with commands)
_CODE_FENCE = re.compile(r"```[\s\S]*?```")

# Whitespace obfuscation (excessive whitespace between characters)
_WHITESPACE_OBFUSCATION = re.compile(r"(\w)\s{3,}(\w)")


def _normalise_homoglyphs(text: str) -> str:
    """Replace known homoglyphs with ASCII equivalents."""
    result = []
    for ch in text:
        result.append(_HOMOGLYPH_MAP.get(ch, ch))
    return "".join(result)


def _try_decode_base64(encoded: str) -> Optional[str]:
    """Try to decode a potential base64 string. Return None if not valid."""
    try:
        decoded = base64.b64decode(encoded + "==").decode("utf-8", errors="strict")
        # Only return if decoded text contains printable ASCII
        if all(32 <= ord(c) <= 126 or c in "\n\r\t" for c in decoded):
            return decoded
    except Exception:
        pass
    return None


class InputSanitizationDefense:
    """
    Aggressive input sanitiser that removes obfuscation techniques
    used to hide injection payloads.

    Removes:
      - Zero-width and invisible Unicode characters
      - Homoglyph substitutions (Cyrillic-for-Latin, etc.)
      - Base64-encoded payloads (decoded and checked for injections)
      - HTML/Markdown comments
      - Unicode control characters (category Cf, Cc)
      - Excessive whitespace obfuscation
    """

    def __init__(self, flag_base64: bool = True):
        """
        Parameters
        ----------
        flag_base64 : whether to flag detected base64 segments
        """
        self.flag_base64 = flag_base64

    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize input text.

        Parameters
        ----------
        text : raw input string

        Returns
        -------
        (cleaned_text, list_of_removed_patterns)
        """
        removed = []
        cleaned = text

        # 1. Remove HTML/Markdown comments
        if _HTML_COMMENT.search(cleaned):
            removed.append("html_comment")
        cleaned = _HTML_COMMENT.sub(" ", cleaned)

        # 2. Remove zero-width / invisible characters
        before = cleaned
        cleaned = _ZERO_WIDTH_CHARS.sub("", cleaned)
        if cleaned != before:
            removed.append("zero_width_chars")

        # 3. Normalise homoglyphs
        before = cleaned
        cleaned = _normalise_homoglyphs(cleaned)
        if cleaned != before:
            removed.append("homoglyph_substitution")

        # 4. Strip Unicode control characters (categories Cf and Cc)
        before = cleaned
        cleaned = "".join(
            c for c in cleaned
            if unicodedata.category(c) not in ("Cf", "Cc")
            or c in ("\n", "\r", "\t")  # keep standard whitespace
        )
        if cleaned != before:
            removed.append("unicode_control_chars")

        # 5. Detect and flag/remove base64 payloads
        if self.flag_base64:
            b64_matches = _BASE64_PATTERN.findall(cleaned)
            for match in b64_matches:
                decoded = _try_decode_base64(match)
                if decoded and len(decoded) > 15:
                    removed.append(f"base64_payload:{match[:20]}...")
                    cleaned = cleaned.replace(match, "[BASE64_REMOVED]")

        # 6. Remove code fence blocks (potential multi-line injections)
        if _CODE_FENCE.search(cleaned):
            removed.append("code_fence_block")
        cleaned = _CODE_FENCE.sub("[CODE_BLOCK_REMOVED]", cleaned)

        # 7. Fix whitespace obfuscation
        before = cleaned
        cleaned = _WHITESPACE_OBFUSCATION.sub(r"\1 \2", cleaned)
        if cleaned != before:
            removed.append("whitespace_obfuscation")

        # 8. Normalize repeated special characters used in GCG-style attacks
        # e.g., "! ! ! AWARD ! ! !" → "AWARD"
        before = cleaned
        cleaned = re.sub(r"(?:!\s*){3,}", "", cleaned)
        cleaned = re.sub(r"(?:\[\[\[|\]\]\])", "", cleaned)
        if cleaned != before:
            removed.append("gcg_special_tokens")

        return cleaned.strip(), removed

    def sanitize_field(self, text: str) -> str:
        """Convenience method — returns only the cleaned text."""
        cleaned, _ = self.sanitize(text)
        return cleaned


# ──────────────────────────────────────────────────────────
# 4. Defense Chain
# ──────────────────────────────────────────────────────────

class DefenseChain:
    """
    Composes multiple defense layers into a single processing pipeline.

    Each defense is applied in order:
      1. InputSanitizationDefense (pre-processing)
      2. SemanticSimilarityDefense (instruction integrity check)
      3. OutputValidationDefense (post-processing, if response provided)

    Parameters
    ----------
    defenses : list of defense instances. Supported types:
               InputSanitizationDefense, SemanticSimilarityDefense,
               OutputValidationDefense
    """

    def __init__(self, defenses: Optional[List[Any]] = None):
        if defenses is None:
            defenses = [
                InputSanitizationDefense(),
                SemanticSimilarityDefense(),
                OutputValidationDefense(),
            ]
        self.defenses = defenses

    def apply(
        self,
        instruction: str,
        data: str,
        response: Optional[str] = None,
        expected_output_type: str = "SELECT_VENDOR",
    ) -> DefenseResult:
        """
        Apply all defense layers to the instruction and data.

        Parameters
        ----------
        instruction          : trusted instruction string
        data                 : untrusted data string (may contain injection)
        response             : optional agent response for output validation
        expected_output_type : for output validation layer

        Returns
        -------
        DefenseResult summarising all layer outputs
        """
        cleaned_instruction = instruction
        cleaned_data = data
        all_flags = []
        all_removed = []
        was_flagged = False
        similarity_score = None
        defense_decisions = {}

        for defense in self.defenses:
            # ── InputSanitizationDefense ──────────────────
            if isinstance(defense, InputSanitizationDefense):
                cleaned_instruction, removed_i = defense.sanitize(instruction)
                cleaned_data, removed_d = defense.sanitize(data)
                all_removed.extend(removed_i)
                all_removed.extend(removed_d)
                if removed_i or removed_d:
                    was_flagged = True
                    all_flags.append("input_sanitization_triggered")
                defense_decisions["input_sanitization"] = {
                    "instruction_changes": removed_i,
                    "data_changes": removed_d,
                }

            # ── SemanticSimilarityDefense ─────────────────
            elif isinstance(defense, SemanticSimilarityDefense):
                score, is_safe = defense.check_similarity(
                    cleaned_instruction, cleaned_data
                )
                similarity_score = score
                # Also check if data itself looks like instructions
                data_score, data_has_instructions = defense.check_data_for_instructions(
                    cleaned_data
                )
                if not is_safe:
                    was_flagged = True
                    all_flags.append(f"low_instruction_similarity:{score:.3f}")
                if data_has_instructions:
                    was_flagged = True
                    all_flags.append(f"data_contains_instructions:{data_score:.3f}")
                defense_decisions["semantic_similarity"] = {
                    "instruction_similarity": score,
                    "is_safe": is_safe,
                    "data_instruction_score": data_score,
                    "data_contains_instructions": data_has_instructions,
                }

            # ── OutputValidationDefense ───────────────────
            elif isinstance(defense, OutputValidationDefense):
                if response is not None:
                    cleaned_resp, resp_flagged = defense.validate_output(
                        response, expected_type=expected_output_type
                    )
                    if resp_flagged:
                        was_flagged = True
                        all_flags.append("output_validation_triggered")
                    defense_decisions["output_validation"] = {
                        "response_flagged": resp_flagged,
                    }

        return DefenseResult(
            cleaned_instruction=cleaned_instruction,
            cleaned_data=cleaned_data,
            is_safe=not was_flagged,
            was_flagged=was_flagged,
            similarity_score=similarity_score,
            flags=all_flags,
            removed_patterns=all_removed,
            defense_decisions=defense_decisions,
        )

    def apply_to_data_only(self, data: str) -> Tuple[str, bool, List[str]]:
        """
        Convenience: apply sanitization + similarity check to data only.

        Returns
        -------
        (cleaned_data, was_flagged, flags)
        """
        result = self.apply(instruction="", data=data)
        return result.cleaned_data, result.was_flagged, result.flags
