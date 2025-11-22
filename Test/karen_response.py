# karen_response.py — Presidential Review Board (Satirical Edition, EN)
# API: class KarenResponder(seed:int=46, tone:str="playful")
# Usage: responder.generate(scores: dict, feats: dict, max_tips: int = 3, external_verdict: str | None = None) -> dict
#
# Output fields (unchanged):
#   - karen_text (str)
#   - verdict (str): "PASS"/"BORDERLINE"/"FAIL"
#   - tags_triggered (list[str])
#   - tags_explained (str)
#   - action_focus (None)
#   - debug (dict)

from typing import Dict, Any, List
import random

# ---- Tier-A keys (display labels kept for potential future use) ----------------
TIER_A_KEYS = [
    "loudness_db",            # dBFS (presence)
    "pause_mean_s",           # s
    "pause_in_target_ratio",  # 0..1
    "speaking_rate_wps",      # words/sec (approx)
    "f0_mean",                # Hz
    "int_var_std_db",         # dB
    "final_f0_drop_st",       # semitones (negative = falling)
    "final_loud_drop_db"      # dB (negative = softer ending)
]

TIER_A_LABEL = {
    "loudness_db": ("Loudness", "dBFS"),
    "pause_mean_s": ("Avg Pause", "s"),
    "pause_in_target_ratio": ("1.0s-Pause Window", ""),
    "speaking_rate_wps": ("Speech Rate", "wps"),
    "f0_mean": ("Pitch Mean", "Hz"),
    "int_var_std_db": ("Intensity Var", "dB"),
    "final_f0_drop_st": ("Final Pitch Δ", "st"),
    "final_loud_drop_db": ("Final Loud Δ", "dB"),
}

# ---- Tag explanations ----------------------------------------------------------
TAG_EXPLAIN = {
    "high_pitch":   "Pitch sits high; reads younger/tenser on a debate stage.",
    "low_pitch":    "Pitch sits low; can read as somber or constrained.",
    "too_fast":     "Tempo exceeds the sweet spot; ideas stack faster than ears digest.",
    "too_slow":     "Cadence drags; momentum and attention leak between clauses.",
    "filler_heavy": "Frequent uh/um/like; padding blurs intent on-air.",
    "weak_cadence": "Phrase endings don’t land; audience can’t find the clap line.",
    "low_loudness": "Projection below target; presence is thin for broadcast.",
    "over_paused":  "Many micro-pauses; flow feels choppy.",
    "under_paused": "Little to no spacing; key points don’t get a beat to resonate.",
    "harsh_bright": "Top-end glare/sibilance; sounds sharp through mics.",
    "jittery":      "Subtle tonal shake; nerves peek through sustained vowels.",
    "weak_ending_loudness": "Endings don’t soften; emotional landing is unclear.",
}

# ================================================================================
# Z-score anchors & helpers
# These make tag selection adaptive (distance from an anchor) rather than hard thresholds.
# You can later re-calibrate (mu, sd) with your own dataset.
# ================================================================================
ANCHORS = {
    "f0_mean": (180.0, 60.0),                 # Hz
    "speaking_rate_wps": (2.2, 0.7),          # words per second
    "pause_in_target_ratio": (0.08, 0.06),    # proportion hitting 0.6–1.2s pauses
    "final_f0_drop_st": (-1.5, 1.0),          # semitone change at ending (neg = falling)
    "final_loud_drop_db": (-1.8, 1.2),        # loudness drop at ending (neg = softer)
}

def _z(name: str, val) -> float | None:
    try:
        mu, sd = ANCHORS.get(name, (0.0, 1.0))
        v = float(val)
        sd = sd if (sd and sd != 0.0) else 1.0
        return (v - mu) / sd
    except Exception:
        return None

def _severity_from_z(zabs: float) -> str:
    # Tunable buckets
    if zabs >= 1.8:
        return "severe"
    elif zabs >= 1.2:
        return "moderate"
    else:
        return "mild"

# ================================================================================
# Simple utilities
# ================================================================================
def _fmt_float(x: float, decimals: int = 2) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return str(x)

def _get_score(scores: Dict[str, float], key: str, default: float = 0.0) -> float:
    if key in scores:
        return float(scores.get(key, default))
    cap = key[0].upper() + key[1:]
    return float(scores.get(cap, default))

def _pick(rng: random.Random, items: List[str]) -> str:
    if not items:
        return ""
    return rng.choice(items)

# ================================================================================
# Witty lines table (severity-aware)
# ================================================================================
WITTY = {
    "too_fast": {
        "mild":     "Great engine, missing commas.",
        "moderate": "You’re auctioning democracy—sold to the fastest sentence.",
        "severe":   "Slow down. The captions are filing a complaint.",
    },
    "too_slow": {
        "mild":     "Give the words some shoes.",
        "moderate": "Filibuster energy—save it for the Senate.",
        "severe":   "Pulse check: the speech needs one.",
    },
    "high_pitch": {
        "mild":     "Bright is fine; helium is a genre.",
        "moderate": "Confidence wants a chest voice, not karaoke night.",
        "severe":   "Land the pitch—less helium, more podium.",
    },
    "low_pitch": {
        "mild":     "Low reads serious; add lift for warmth.",
        "moderate": "Add a notch of light; gravity isn’t a brand.",
        "severe":   "Voice is tunneling—raise the ceiling a touch.",
    },
    "under_paused": {
        "mild":     "Let nouns breathe; voters inhale, too.",
        "moderate": "Zero chill between ideas—grant one heartbeat.",
        "severe":   "Space sells the message—build a balcony.",
    },
    "over_paused": {
        "mild":     "Too many stop signs—merge a few lanes.",
        "moderate": "Punctuation is winning by landslide.",
        "severe":   "Chop less, carry more.",
    },
    "weak_cadence": {
        "mild":     "Put a period on it; we’ll supply the applause.",
        "moderate": "Endings fade like a promise the day after.",
        "severe":   "The clap line can’t find you—leave a sign.",
    },
    "weak_ending_loudness": {
        "mild":     "Let the ending exhale.",
        "moderate": "Soften the landing so meaning sticks.",
        "severe":   "Endings collide with the wall—feather the brakes.",
    },
    "jittery": {
        "mild":     "Nerves waving hello—wave back, then speak.",
        "moderate": "Tone is shaking hands with itself—steady the greeting.",
        "severe":   "Vibes vibrate; give them a chair.",
    },
    "low_loudness": {
        "mild":     "Turn the porch light on.",
        "moderate": "Good platform—add batteries to the megaphone.",
        "severe":   "Mic hears a secret; the crowd hears a rumor.",
    },
}

def _witty_for(tag: str, severity: str) -> str:
    return WITTY.get(tag, {}).get(severity, "")

# ================================================================================
# Tag selection (z-score based, adaptive)
# Returns (tags:list[str], info:list[dict])
# ================================================================================
def select_tags_z(feats: dict, max_tips: int = 3):
    info = []

    def add(tag: str, zval: float, note: str = ""):
        info.append({
            "tag": tag,
            "z": zval,
            "severity": _severity_from_z(abs(zval)),
            "note": note
        })

    # Speaking rate
    z_rate = _z("speaking_rate_wps", feats.get("speaking_rate_wps"))
    if z_rate is not None:
        if z_rate >= 1.3:
            add("too_fast", z_rate, "Rate above anchor")
        elif z_rate <= -1.0:
            add("too_slow", z_rate, "Rate below anchor")

    # Mean pitch
    z_f0 = _z("f0_mean", feats.get("f0_mean"))
    if z_f0 is not None:
        if z_f0 >= 1.2:
            add("high_pitch", z_f0, "Pitch above anchor")
        elif z_f0 <= -1.1:
            add("low_pitch", z_f0, "Pitch below anchor")

    # Pause ratio (target 0.6–1.2s)
    z_pause = _z("pause_in_target_ratio", feats.get("pause_in_target_ratio"))
    if z_pause is not None:
        if z_pause <= -1.0:
            add("under_paused", z_pause, "Few target pauses")
        elif z_pause >= 1.2:
            add("over_paused", z_pause, "Many target pauses")

    # Ending cadence (pitch drop)
    z_final_f0 = _z("final_f0_drop_st", feats.get("final_f0_drop_st"))
    if z_final_f0 is not None and z_final_f0 >= 0.7:
        add("weak_cadence", z_final_f0, "Ending pitch drop insufficient")

    # Ending loudness drop
    z_final_loud = _z("final_loud_drop_db", feats.get("final_loud_drop_db"))
    if z_final_loud is not None and z_final_loud >= 0.7:
        add("weak_ending_loudness", z_final_loud, "Ending loudness drop insufficient")

    # Optional: jitter as a simple hard threshold (no anchor in this file yet)
    try:
        if float(feats.get("jitter", 0.01)) > 0.02:
            add("jittery", 2.0, "Jitter above heuristic threshold")  # fake z for severity buckets
    except Exception:
        pass

    # Deduplicate by tag, keep strongest (largest |z|)
    strongest = {}
    for t in info:
        key = t["tag"]
        if key not in strongest or abs(t["z"]) > abs(strongest[key]["z"]):
            strongest[key] = t
    ordered = sorted(strongest.values(), key=lambda d: abs(d["z"]), reverse=True)
    clipped = ordered[:max_tips]

    return [d["tag"] for d in clipped], clipped

# ================================================================================
# Responder
# ================================================================================
class KarenResponder:
    def __init__(self, seed: int = 46, tone: str = "playful"):
        self.rng = random.Random(seed)
        self.tone = tone

        self.verdict_bank = {
            "pass": [
                "Electability: 🟢 PASSED. The room nods, the cameras hum—America can hear you.",
                "Electability: 🟢 PASSED. Message delivered; we almost drafted a slogan.",
                "Electability: 🟢 PASSED. Production approves, democracy withholds comment.",
            ],
            "borderline": [
                "Electability: 🟡 BORDERLINE. The blueprint holds; the furniture needs rearranging.",
                "Electability: 🟡 BORDERLINE. Almost broadcast-ready—trim the static, keep the spark.",
                "Electability: 🟡 BORDERLINE. The pitch is playable; tune two strings.",
            ],
            "fail": [
                "Electability: 🔴 NOT YET. The ideas exist; the delivery denies them visas.",
                "Electability: 🔴 NOT YET. Stagecraft wobbles—rehearse the spine, then the swagger.",
                "Electability: 🔴 NOT YET. Great draft, wrong microphone.",
            ],
        }

    # Fallback verdict (kept for compatibility if external_verdict is not provided)
    def _verdict_key(self, scores: Dict[str, float]) -> str:
        A = _get_score(scores, "authority")
        T = _get_score(scores, "trust")
        C = _get_score(scores, "clarity")
        F = _get_score(scores, "fluency")
        passed = (A >= 0.60) and (T >= 0.55) and (C >= 0.55) and (F >= 0.55)
        borderline = (A >= 0.50) and (T >= 0.48) and (C >= 0.48) and (F >= 0.48)
        return "pass" if passed else ("borderline" if borderline else "fail")

    # (Kept; not currently printed by generate to avoid duplicating main script output)
    def _tier_a_dump(self, feats: Dict[str, Any]) -> str:
        parts = []
        for key in TIER_A_KEYS:
            label, unit = TIER_A_LABEL.get(key, (key, ""))
            val = feats.get(key, None)
            if val is None:
                continue
            if key in ("pause_in_target_ratio",):
                s = f"{label}: {_fmt_float(100.0*float(val), 1)}%"
            else:
                s = f"{label}: {_fmt_float(val, 2)}{(' ' + unit) if unit else ''}"
            parts.append(s)
        return " | ".join(parts)

    def _scores_dump(self, scores: Dict[str, float]) -> str:
        labels = [("Authority","authority"),("Trust","trust"),
                  ("Clarity","clarity"),("Fluency","fluency"),("Warmth","warmth")]
        parts = [f"{L}: {_fmt_float(_get_score(scores, k), 2)}" for (L,k) in labels]
        parts.append(f"Cadence: {_fmt_float(_get_score(scores, 'cadence'), 2)}")
        return " | ".join(parts)

    def _explain_tags_block(self, tags: List[str]) -> str:
        if not tags:
            return "Triggered tags: (none)"
        items = []
        for t in tags:
            desc = TAG_EXPLAIN.get(t, "—")
            items.append(f"{t}: {desc}")
        return "Triggered tags: " + "; ".join(items)

    def _severity_witty_lines(self, tag_info: List[dict], k: int) -> List[str]:
        lines: List[str] = []
        for t in tag_info[:k]:
            w = _witty_for(t["tag"], t["severity"])
            if w:
                lines.append(w)
        return lines

    def generate(
        self,
        scores: Dict[str, float],
        feats: Dict[str, Any],
        max_tips: int = 3,
        external_verdict: str | None = None
    ) -> Dict[str, Any]:

        # 1) Verdict: prefer external_verdict from the main script to stay in sync
        if external_verdict in ("PASS", "BORDERLINE", "FAIL"):
            vkey = external_verdict.lower()
        else:
            vkey = self._verdict_key(scores)

        verdict_line = _pick(self.rng, self.verdict_bank[vkey])

        # 2) Tags via z-score logic (adaptive) + severity-aware witty lines
        tags, tag_info = select_tags_z(feats, max_tips=max_tips)
        tag_lines = self._severity_witty_lines(tag_info, k=max_tips)

        # 3) Compose (keep compact; main script already prints Tier-A & scores)
        header = "Presidential Review Board (satirical):"
        explained = self._explain_tags_block(tags)

        body_lines = [
            header,
            verdict_line,
            *tag_lines,
            explained,
        ]
        body = "\n".join([ln for ln in body_lines if ln])

        return {
            "karen_text": body,
            "verdict": vkey.upper(),
            "tags_triggered": tags,
            "tags_explained": explained,
            "action_focus": None,
            "debug": {
                "scores_keys_seen": list(scores.keys())[:8],
                "feats_sample": {k: feats.get(k) for k in ["f0_mean", "speaking_rate_wps", "jitter",
                                                           "pause_in_target_ratio", "final_f0_drop_st",
                                                           "final_loud_drop_db"]},
                "tag_info": tag_info,
            },
        }
