# karen_score_tiered.py
# Usage:
#   python karen_score_tiered.py "path/to/audio.wav|m4a|mp3"
# Dependencies: numpy, librosa
import os, sys, json, math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import librosa

# ------------ Config ------------
SR = 16000               # unified sampling rate
FRAME_HOP = 0.02         # 20ms hop
FRAME_LEN = 0.04         # 40ms window
HOP_LENGTH = int(SR * FRAME_HOP)
N_FFT = 2048

# === Core vs Support (Tiering) ===
TIER_A_KEYS = {
    "loudness_db",                 # INT-DB
    "pause_mean_s",                # DUR-PSE
    "pause_in_target_ratio",       # DUR-PSE
    "speaking_rate_wps",           # RATE-SP (word/s approx)
    "f0_mean",                     # F0
    "int_var_std_db",              # INT-VAR
    "final_f0_drop_st",            # END-CAD
    "final_loud_drop_db"           # END-CAD
}

# Weights (linear -> sigmoid)
WEIGHTS = {
    "authority": {
        "f0_mean": -0.45,
        "f0_std": -0.10,
        "loudness_db": 0.35,
        "pause_ratio": -0.10,
        "pause_long_ratio": 0.28,
        "filler_like_ratio": -0.35,
        "jitter": -0.20,
        # new
        "pause_in_target_ratio": 0.16,
        "final_f0_drop_st": -0.18,     # more negative (fall) => better
        "final_loud_drop_db": -0.12,   # more negative (fall) => better
        "int_var_std_db": 0.10
    },
    "trust": {
        "jitter": -0.28,
        "shimmer": -0.20,
        "clarity": 0.30,
        "speaking_rate_wps": 0.12,
        "noise_floor_db": -0.08,
        "filler_like_ratio": -0.30,
        "pause_cv": -0.08,
        # new
        "pause_in_target_ratio": 0.08,
        "final_f0_drop_st": -0.08
    },
    "clarity": {
        "clarity": 0.45,
        "spectral_rolloff": -0.05,
        "spectral_centroid": -0.05,
        "pause_micro_ratio": -0.08,
        "pause_ratio": -0.06
    },
    "fluency": {
        "pause_ratio": -0.18,
        "speaking_rate_wps": 0.22,
        "f0_std": -0.05,
        "filler_like_ratio": -0.40,
        "pause_long_ratio": 0.10,
        # new
        "pause_in_target_ratio": 0.14
    },
    "warmth": {
        "f0_mean": -0.12,
        "loudness_db": 0.10,
        "spectral_centroid": -0.10,
        # new
        "int_var_std_db": 0.08
    }
}

BIASES = {  # per-dimension bias (intercept)
    "authority": 0.0,
    "trust": 0.0,
    "clarity": 0.0,
    "fluency": 0.0,
    "warmth": 0.0
}

PASS_RULE = {  # pass conditions
    "authority": 0.55,
    "trust": 0.50,
    "clarity": 0.50,
    "fluency": 0.50,
}

# Snarky/encouraging comments
KAREN_SNARKS = [
    # negative triggers
    (lambda f,s: f.get("f0_mean",0) > 200 and s["authority"] < 0.55, "High pitch gives me intern vibes."),
    (lambda f,s: f.get("filler_like_ratio",0) > 0.04, "The little 'uh' and 'um'—like pebbles in your shoes."),
    (lambda f,s: f.get("pause_micro_ratio",0) > 0.08 and s["fluency"] < 0.5, "Choppy rhythm. This isn’t a buffering icon."),
    (lambda f,s: f.get("loudness_db",-30) < -18, "Speak up. This is a rally, not a library."),
    (lambda f,s: f.get("jitter",0) > 0.03, "Your voice shakes like my iced latte."),
    (lambda f,s: s["trust"] < 0.4 and s["clarity"] < 0.5, "Hard to trust what I can barely parse."),
    # positive triggers
    (lambda f,s: f.get("pause_long_ratio",0) > 0.06 and s["authority"] > 0.6, "Those deliberate pauses? Presidential."),
    (lambda f,s: s["clarity"] > 0.65 and s["fluency"] > 0.6, "Clean articulation and steady flow—your message lands."),
    (lambda f,s: s["warmth"] > 0.6 and s["trust"] > 0.55, "You sound like someone people want to follow, not just hear."),
    (lambda f,s: s["authority"] > 0.7, "Commanding presence. The room moves when you do.")
]

# ------------ Utils ------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def contiguous_runs(mask: np.ndarray):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j+1 < n and mask[j+1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

def rms_db(y: np.ndarray) -> float:
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def moving_energy(y: np.ndarray, frame_len: int, hop_len: int):
    if len(y) < frame_len: 
        return np.array([np.mean(y**2)])
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    return np.mean(frames**2, axis=0)

def estimate_pitch(y: np.ndarray, sr: int) -> Tuple[float,float,float]:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=int(sr * 0.06),
        hop_length=HOP_LENGTH
    )
    f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0v) == 0:
        return 0.0, 0.0, 0.0
    f0_mean = float(np.mean(f0v))
    f0_std = float(np.std(f0v))
    # simple jitter proxy
    df = np.abs(np.diff(f0v))
    jitter = float(np.mean(df / (f0v[:-1] + 1e-9))) if len(df) > 0 else 0.0
    return f0_mean, f0_std, jitter

def estimate_shimmer(y: np.ndarray, sr: int) -> float:
    # frame RMS and relative change median as shimmer proxy
    frame_len = int(sr*FRAME_LEN)
    hop_len = HOP_LENGTH
    if len(y) < frame_len:
        return 0.0
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    amp = np.sqrt(np.mean(frames**2, axis=0)) + 1e-9
    return float(np.median(np.abs(np.diff(amp)) / (amp[:-1] + 1e-9))) if len(amp) > 1 else 0.0

def db_track(y: np.ndarray, sr: int, win_s=0.25, hop_s=0.125):
    win = int(sr * win_s); hop = int(sr * hop_s)
    if win <= 0 or hop <= 0 or len(y) < win:
        return np.array([])
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop)
    rms = np.sqrt(np.mean(frames**2, axis=0)) + 1e-12
    return 20 * np.log10(rms)

def compute_intensity_variation(db_seq: np.ndarray) -> Dict[str, float]:
    if db_seq.size == 0:
        return {"int_var_std_db": 0.0}
    return {"int_var_std_db": float(np.std(db_seq))}

def compute_final_cadence(f0_seq: np.ndarray, db_seq: np.ndarray, last_ratio=0.2) -> Dict[str, float]:
    n = min(len(f0_seq), len(db_seq))
    if n < 10:
        return {"final_f0_drop_st": 0.0, "final_loud_drop_db": 0.0}
    cut = int((1.0 - last_ratio) * n)
    f0_prev = np.nanmean(f0_seq[:cut]) if np.any(~np.isnan(f0_seq[:cut])) else np.nan
    f0_end  = np.nanmean(f0_seq[cut:])  if np.any(~np.isnan(f0_seq[cut:]))  else np.nan
    if np.isnan(f0_prev) or np.isnan(f0_end) or f0_prev <= 0 or f0_end <= 0:
        f0_drop_st = 0.0
    else:
        f0_drop_st = 12.0 * np.log2(f0_end / f0_prev)  # negative is a fall
    db_drop = float(np.mean(db_seq[cut:]) - np.mean(db_seq[:cut])) if len(db_seq)>0 else 0.0
    return {"final_f0_drop_st": float(f0_drop_st), "final_loud_drop_db": db_drop}

def spectral_features(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) + 1e-12
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    return centroid, rolloff

def estimate_noise_floor_db(y: np.ndarray, sr: int) -> float:
    frame_len = int(sr*FRAME_LEN)
    hop_len = HOP_LENGTH
    if len(y) < frame_len:
        return -60.0
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    e = np.mean(frames**2, axis=0)
    k = max(1, int(0.1 * len(e)))
    floor = np.mean(np.sort(e)[:k]) + 1e-12
    return 10.0 * math.log10(floor)

def estimate_clarity(y: np.ndarray, sr: int) -> float:
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(sr*FRAME_LEN), hop_length=HOP_LENGTH)[0]
    centroid, rolloff = spectral_features(y, sr)
    zcr_norm = 1.0 - float(np.clip((zcr.mean() - 0.05) / 0.15, 0, 1))
    hf_norm  = 1.0 - float(np.clip((centroid - 2500.0) / 2500.0, 0, 1))
    return float(np.clip(0.5*zcr_norm + 0.5*hf_norm, 0, 1))

def estimate_speaking_rate(y: np.ndarray, sr: int) -> float:
    # very crude proxy: zcr peaks per second -> clamp to [0.8, 4.0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(sr*FRAME_LEN), hop_length=int(sr*FRAME_HOP))[0]
    peaks = (zcr[1:-1] > zcr[:-2]) & (zcr[1:-1] > zcr[2:]) & (zcr[1:-1] > (np.mean(zcr) + np.std(zcr)))
    approx_units_per_sec = peaks.sum() / (len(zcr)/ (sr*FRAME_HOP))
    return float(np.clip(approx_units_per_sec, 0.8, 4.0))

def estimate_pause_features(y: np.ndarray, sr: int):
    """
    Returns:
      pause_ratio, pause_long_ratio, pause_micro_ratio,
      filler_like_ratio, pause_cv,
      pause_mean_s, pause_in_target_ratio
    """
    frame_len = int(sr * FRAME_LEN)
    hop_len   = HOP_LENGTH
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    energy = np.mean(frames**2, axis=0)
    med = np.median(energy) + 1e-9
    thr = med * 0.25
    is_sil = energy < thr

    # voiced proxy
    f0, _, _ = librosa.pyin(y,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=frame_len, hop_length=hop_len)
    is_voiced = ~np.isnan(f0)

    # filler-like: voiced & low-energy & short (80–300 ms) with stable f0
    low_energy = energy < (med * 0.5)
    f0_stable = np.zeros_like(is_voiced, dtype=bool)
    f0v = np.copy(f0)
    f0v[np.isnan(f0v)] = 0.0
    if len(f0v) > 5:
        df = np.abs(np.diff(f0v))
        df = np.concatenate([[0.0], df])
        f0_stable = df < 2.0

    # pauses as contiguous low-energy runs
    runs = contiguous_runs(is_sil)
    ms_per_hop = (hop_len / sr) * 1000.0
    def dur_ms(a,b): return (b - a + 1) * ms_per_hop
    pause_durs = [dur_ms(a,b) for (a,b) in runs]
    total_time_s = len(energy) * (hop_len / sr) + 1e-9
    pause_time_s = sum(pause_durs) / 1000.0

    long_thr = 350.0
    micro_lo, micro_hi = 50.0, 250.0
    long_time_s  = sum(d for d in pause_durs if d >= long_thr) / 1000.0
    micro_time_s = sum(d for d in pause_durs if micro_lo <= d <= micro_hi) / 1000.0

    filler_mask = (~is_sil) & is_voiced & low_energy & f0_stable
    filler_runs = contiguous_runs(filler_mask)
    filler_time_s = 0.0
    for (a,b) in filler_runs:
        d = dur_ms(a,b)
        if 80.0 <= d <= 300.0:
            filler_time_s += d / 1000.0

    pause_ratio        = float(pause_time_s / total_time_s)
    pause_long_ratio   = float(long_time_s  / total_time_s)
    pause_micro_ratio  = float(micro_time_s / total_time_s)
    filler_like_ratio  = float(filler_time_s / total_time_s)
    pause_cv = float(np.std(pause_durs)/ (np.mean(pause_durs)+1e-9)) if pause_durs else 0.0

    # new: mean pause & in-target ratio (0.6–1.2 s)
    pause_mean_s = float(np.mean(pause_durs)/1000.0) if pause_durs else 0.0
    in_lo, in_hi = 600.0, 1200.0
    in_target_s = sum(d for d in pause_durs if in_lo <= d <= in_hi) / 1000.0
    pause_in_target_ratio = float(in_target_s / total_time_s)

    return {
        "pause_ratio": pause_ratio,
        "pause_long_ratio": pause_long_ratio,
        "pause_micro_ratio": pause_micro_ratio,
        "filler_like_ratio": filler_like_ratio,
        "pause_cv": pause_cv,
        "pause_mean_s": pause_mean_s,
        "pause_in_target_ratio": pause_in_target_ratio
    }

def standardize_feature(name: str, val: float) -> float:
    anchors = {
        "f0_mean": (180, 60),               # Hz
        "f0_std": (35, 20),                 # Hz
        "jitter": (0.02, 0.02),             # ratio
        "shimmer": (0.08, 0.05),            # ratio
        "loudness_db": (-15, 6),            # dBFS
        "pause_ratio": (0.15, 0.15),        # 0~1
        "speaking_rate_wps": (2.2, 0.7),    # words/s approx
        "spectral_centroid": (2500, 1200),  # Hz
        "spectral_rolloff": (6000, 2500),   # Hz
        "noise_floor_db": (-40, 8),         # dB
        "clarity": (0.6, 0.2),              # 0~1
        # new anchors
        "pause_mean_s": (0.9, 0.3),             # target ~0.9s
        "pause_in_target_ratio": (0.08, 0.06),  # ratio of time in 0.6–1.2s pauses
        "int_var_std_db": (8.0, 3.0),           # healthy dynamics ~6–10 dB
        "final_f0_drop_st": (-1.5, 1.0),        # semitones (more negative better)
        "final_loud_drop_db": (-1.8, 1.2)       # dB (more negative better)
    }
    mu, sd = anchors.get(name, (0.0, 1.0))
    if sd <= 0: sd = 1.0
    return float((val - mu) / sd)

def score_dimensions(feats: Dict[str,float]) -> Dict[str,float]:
    # z-score features
    xz = {k: standardize_feature(k, v) for k, v in feats.items()}
    dim_scores = {}
    for dim, wmap in WEIGHTS.items():
        z = BIASES.get(dim, 0.0)
        for fname, w in wmap.items():
            z += w * xz.get(fname, 0.0)
        dim_scores[dim] = float(sigmoid(z))
    return dim_scores

def make_karen_comment(feats: Dict[str,float], dim_scores: Dict[str,float]) -> str:
    pos, neg = [], []
    for cond, text in KAREN_SNARKS:
        try:
            if cond(feats, dim_scores):
                if any(k in text for k in ["Presidential", "Clean", "Commanding"]):
                    pos.append(text)
                else:
                    neg.append(text)
        except Exception:
            continue
    if neg: return neg[0]
    if pos: return pos[0]
    return "Nice delivery. Keep the cadence and clarity."

@dataclass
class ProfileCard:
    features: Dict[str,float]
    scores: Dict[str,float]
    decision: str
    karen_text: str

def pass_fail(scores: Dict[str,float]) -> str:
    for k, thr in PASS_RULE.items():
        if scores.get(k, 0.0) < thr:
            return "FAIL"
    return "PASS"

def print_ui(scores, feats, decision, karen_text):
    print("\n[ Calibrating Karen… ]")
    print("▮" * 10)
    print(f"\nAuthority : {scores['authority']:.2f}")
    print(f"Trust     : {scores['trust']:.2f}")
    print(f"Clarity   : {scores['clarity']:.2f}")
    print(f"Fluency   : {scores['fluency']:.2f}")
    print(f"Warmth    : {scores['warmth']:.2f}")
    print("-" * 72)
    print(f"Decision: {'🟢 PASS' if decision=='PASS' else '🔴 FAIL'}")
    print("\nKaren says:")
    print(karen_text)

def split_by_tier(feats: Dict[str,float]):
    A = {k: v for k, v in feats.items() if k in TIER_A_KEYS}
    B = {k: v for k, v in feats.items() if k not in TIER_A_KEYS}
    return A, B

def main():
    audio_path = sys.argv[1] if len(sys.argv) >= 2 else None
    if not audio_path:
        print("Usage: python karen_score_tiered.py path/to/audio.wav")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"[Error] Audio not found: {audio_path}")
        sys.exit(1)

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    if len(y)==0:
        print("[Error] Empty audio.")
        sys.exit(1)
    y = librosa.util.normalize(y)

    # --- features ---
    f0_mean, f0_std, jitter = estimate_pitch(y, SR)
    loud_db = rms_db(y)
    energy = moving_energy(y, int(SR*FRAME_LEN), HOP_LENGTH)  # not directly used but available
    pause_feats = estimate_pause_features(y, SR)
    speaking_rate = estimate_speaking_rate(y, SR)
    centroid, rolloff = spectral_features(y, SR)
    shimmer = estimate_shimmer(y, SR)
    clarity = estimate_clarity(y, SR)
    noise_db = estimate_noise_floor_db(y, SR)

    # tracks for INT-VAR / END-CAD
    f0_seq, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=int(SR*0.06), hop_length=HOP_LENGTH
    )
    db_seq = db_track(y, SR, win_s=0.25, hop_s=0.125)
    intvar = compute_intensity_variation(db_seq)
    endcad = compute_final_cadence(f0_seq, db_seq, last_ratio=0.2)

    feats = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "loudness_db": loud_db,
        "pause_ratio": pause_feats["pause_ratio"],
        "speaking_rate_wps": speaking_rate,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "noise_floor_db": noise_db,
        "clarity": clarity,
        "pause_long_ratio": pause_feats["pause_long_ratio"],
        "pause_micro_ratio": pause_feats["pause_micro_ratio"],
        "filler_like_ratio": pause_feats["filler_like_ratio"],
        "pause_cv": pause_feats["pause_cv"],

        # Tier-A direct measures
        "pause_mean_s": pause_feats["pause_mean_s"],
        "pause_in_target_ratio": pause_feats["pause_in_target_ratio"],
        "int_var_std_db": intvar["int_var_std_db"],
        "final_f0_drop_st": endcad["final_f0_drop_st"],
        "final_loud_drop_db": endcad["final_loud_drop_db"],
    }

    scores = score_dimensions(feats)
    decision = pass_fail(scores)
    karen_text = make_karen_comment(feats, scores)

    print_ui(scores, feats, decision, karen_text)

    # assemble payload
    A, B = split_by_tier(feats)
    card = ProfileCard(features=feats, scores=scores, decision=decision, karen_text=karen_text)
    out_dir = os.path.join(os.path.dirname(audio_path), "Output")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "karen_result_tiered.json")

    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    payload = _to_py(asdict(card))
    payload["features"] = {"tierA_core": A, "tierB_support": B}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
