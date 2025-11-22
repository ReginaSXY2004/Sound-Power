# karen_score_tiered_v2.py
# Usage (Windows PowerShell / cmd):
#   python karen_score_tiered_v2.py "C:\\path\\to\\audio.wav"
# Or set DEFAULT_AUDIO below and just run: python karen_score_tiered_v2.py
#
# What it saves:
#   1) JSON:  <audio_dir>/Output/karen_result_tiered.json
#   2) CSV:   <audio_dir>/Output/karen_runs.csv   (A_/B_/S_ columns)
#
# Dependencies: numpy, librosa

import os, sys, csv, json, math, datetime, re
sys.path.insert(0, os.path.dirname(__file__))  # 让Python能看到当前脚本所在目录
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
from karen_response import KarenResponder # 新增行
# 1) 打印实际加载到的 karen_response 路径（定位是否跑错文件）
import karen_response as _kr
print("[DEBUG] karen_response loaded from:", _kr.__file__)

import numpy as np
import librosa

# ===== Terminal colors & pretty score =====
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def color_score(val: float) -> str:
    """Return colored string for a score value."""
    if val >= 0.70:
        return f"{GREEN}{val:.2f}{RESET}"
    elif val >= 0.50:
        return f"{YELLOW}{val:.2f}{RESET}"
    else:
        return f"{RED}{val:.2f}{RESET}"

# ----------------- Config -----------------
SR = 16000               # unified sampling rate
FRAME_HOP = 0.02         # 20ms hop
FRAME_LEN = 0.04         # 40ms window
HOP_LENGTH = int(SR * FRAME_HOP)
N_FFT = 2048

# Optional: set a default audio path here if you don't want to pass CLI arg
DEFAULT_AUDIO = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\TestAudioInput\Obama.wav"  

#Core vs Support Tiering
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
        "f0_mean": -0.35,
        "f0_std": -0.10,
        "loudness_db": 0.35,
        "pause_ratio": -0.10,
        "pause_long_ratio": 0.28,
        "filler_like_ratio": -0.35,
        "jitter": -0.20,
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
        "pause_in_target_ratio": 0.14
    },
    "warmth": {
        "f0_mean": -0.12,
        "loudness_db": 0.10,
        "spectral_centroid": -0.10,
        "int_var_std_db": 0.08
    },
    "cadence": {
        "final_f0_drop_st": -0.35, 
        "final_loud_drop_db": -0.25, 
        "pause_in_target_ratio": 0.60, 
        "pause_mean_s": 0.20
    }
}

BIASES = {  # per-dimension bias (intercept)
    "authority": 0.0,
    "trust": 0.0,
    "clarity": 0.0,
    "fluency": 0.0,
    "warmth": 0.0,
    "cadence": 0.0
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

# ----------------- Utils -----------------
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

    # 太短直接返回0集
    if len(y) < frame_len:
        return {
            "pause_ratio": 0.0, "pause_long_ratio": 0.0, "pause_micro_ratio": 0.0,
            "filler_like_ratio": 0.0, "pause_cv": 0.0,
            "pause_mean_s": 0.0, "pause_in_target_ratio": 0.0
        }

    # 能量帧（不center，和 util.frame 一致）
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    energy = np.mean(frames**2, axis=0)
    med = np.median(energy) + 1e-9
    thr = med * 0.25
    is_sil = energy < thr
    low_energy = energy < (med * 0.5)

    # f0 帧（pyin），用于“有声/稳定”判定
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=frame_len, hop_length=hop_len
    )
    is_voiced = ~np.isnan(f0)

    # 以帧差衡量F0稳定
    f0v = np.copy(f0)
    f0v[np.isnan(f0v)] = 0.0
    if len(f0v) > 5:
        df = np.abs(np.diff(f0v))
        df = np.concatenate([[0.0], df])  # 对齐长度
        f0_stable = df < 2.0
    else:
        f0_stable = np.zeros_like(is_voiced, dtype=bool)

    # —— 关键修复：统一长度 —— #
    L = min(len(is_sil), len(low_energy), len(is_voiced), len(f0_stable), len(energy))
    is_sil     = is_sil[:L]
    low_energy = low_energy[:L]
    is_voiced  = is_voiced[:L]
    f0_stable  = f0_stable[:L]
    energy     = energy[:L]

    # 之后再做停顿连段与统计
    ms_per_hop = (hop_len / sr) * 1000.0
    def dur_ms(a,b): return (b - a + 1) * ms_per_hop

    # 静音区段（停顿）
    runs = contiguous_runs(is_sil)
    pause_durs = [dur_ms(a,b) for (a,b) in runs]
    total_time_s = len(energy) * (hop_len / sr) + 1e-9
    pause_time_s = sum(pause_durs) / 1000.0

    # 长停顿 / 微停顿
    long_thr = 350.0
    micro_lo, micro_hi = 50.0, 250.0
    long_time_s  = sum(d for d in pause_durs if d >= long_thr) / 1000.0
    micro_time_s = sum(d for d in pause_durs if micro_lo <= d <= micro_hi) / 1000.0

    # 类“呃/嗯”填充段：有声 & 低能量 & f0 稳定，且 80–300ms
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

    # 六参的对表量：平均停顿 & 目标窗命中
    pause_mean_s = float(np.mean(pause_durs)/1000.0) if pause_durs else 0.0
    in_lo, in_hi = 600.0, 1200.0  # ms
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
        "f0_mean": (180, 40),               # Hz
        "f0_std": (35, 20),                 # Hz
        "jitter": (0.02, 0.02),             # ratio
        "shimmer": (0.08, 0.05),            # ratio
        "loudness_db": (-15, 6),            # dBFS
        "pause_ratio": (0.15, 0.15),        # 0~1
        "speaking_rate_wps": (2.2, 0.5),    # words/s approx
        "spectral_centroid": (2500, 1200),  # Hz
        "spectral_rolloff": (6000, 2500),   # Hz
        "noise_floor_db": (-40, 8),         # dB
        "clarity": (0.6, 0.2),              # 0~1
        "pause_mean_s": (0.9, 0.3),             # target ~0.9s
        "pause_in_target_ratio": (0.08, 0.04),  # ratio of time in 0.6–1.2s pauses
        "int_var_std_db": (8.0, 3.0),           # healthy dynamics ~6–10 dB
        "final_f0_drop_st": (-1.5, 1.0),        # semitones (more negative better)
        "final_loud_drop_db": (-1.8, 1.2)       # dB (more negative better)
    }
    mu, sd = anchors.get(name, (0.0, 1.0))
    if sd <= 0: sd = 1.0
    z = (val - mu) / sd
    z = max(min(z, 3.0), -3.0)
    return float(z)


def score_dimensions(feats: Dict[str, float]) -> Dict[str, float]:
    xz = {k: standardize_feature(k, v) for k, v in feats.items()}
    dim_scores = {}
    for dim, wmap in WEIGHTS.items():
        z = BIASES.get(dim, 0.0)
        for fname, w in wmap.items():
            z += w * xz.get(fname, 0.0)
        z *= 1.8  # amplify contrast
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

def pass_fail(scores):
    dims = ["authority","trust","clarity","fluency","warmth","cadence"]
    vals = [float(scores.get(k, 0.0)) for k in dims]
    mean_score = float(np.mean(vals))
    if mean_score >= 0.7 and min(vals) >= 0.45:
        return "PASS"
    elif mean_score >= 0.55:
        return "BORDERLINE"
    else:
        return "FAIL"



def _tierA_line(feats: dict) -> str:
    """Format Tier-A features into one readable line."""
    def f(k, label, unit="", pct=False, nd=2):
        v = feats.get(k, None)
        if v is None: 
            return None
        if pct:
            return f"{label}: {100.0*float(v):.1f}%"
        return f"{label}: {float(v):.{nd}f}{(' ' + unit) if unit else ''}"

    parts = [
        f("loudness_db", "Loudness", "dBFS"),
        f("pause_mean_s", "Avg Pause", "s"),
        f("pause_in_target_ratio", "1.0s-Pause Window", pct=True),
        f("speaking_rate_wps", "Speech Rate", "wps"),
        f("f0_mean", "Pitch Mean", "Hz"),
        f("int_var_std_db", "Intensity Var", "dB"),
        f("final_f0_drop_st", "Final Pitch Δ", "st"),
        f("final_loud_drop_db", "Final Loud Δ", "dB"),
    ]
    return " | ".join([p for p in parts if p])

def _tierA_block(feats: dict) -> str:
    rows = []
    def add(label, val, unit="", pct=False, nd=2):
        if val is None: return
        if pct:
            rows.append(f"  - {label}: {100.0*float(val):.{1}f}%")
        else:
            rows.append(f"  - {label}: {float(val):.{nd}f}{(' ' + unit) if unit else ''}")

    add("Loudness", feats.get("loudness_db"), "dBFS")
    add("Avg Pause", feats.get("pause_mean_s"), "s")
    add("1.0s-Pause Window", feats.get("pause_in_target_ratio"), pct=True)
    add("Speech Rate", feats.get("speaking_rate_wps"), "wps")
    add("Pitch Mean", feats.get("f0_mean"), "Hz")
    add("Intensity Var", feats.get("int_var_std_db"), "dB")
    add("Final Pitch \u0394", feats.get("final_f0_drop_st"), "st")
    add("Final Loud \u0394", feats.get("final_loud_drop_db"), "dB")
    return "\n".join(rows)

def _scores_block(scores: dict) -> str:
    def c(v):
        # 颜色阈值：>=0.7 绿，0.5~0.7 黄，<0.5 红
        GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"
        if v >= 0.7:   return f"{GREEN}{v:.2f}{RESET}"
        if v >= 0.5:   return f"{YELLOW}{v:.2f}{RESET}"
        return f"{RED}{v:.2f}{RESET}"
    lines = [
        f"  - Authority: {c(scores.get('authority',0.0))}",
        f"  - Trust    : {c(scores.get('trust',0.0))}",
        f"  - Clarity  : {c(scores.get('clarity',0.0))}",
        f"  - Fluency  : {c(scores.get('fluency',0.0))}",
        f"  - Warmth   : {c(scores.get('warmth',0.0))}",
        f"  - Cadence  : {c(scores.get('cadence',0.0))}",
    ]
    return "\n".join(lines)






def save_full_report_text(scores, feats, decision, response,
                          out_path=r"Output\karen_report.txt"):
    import numpy as np, os, re

    PASS_DIMS = ["authority","trust","clarity","fluency","warmth","cadence"]

    lines = []
    # 标题
    lines.append("[ Presidential Review Board (sim.) ]")
    lines.append("")
    # Tier-A block
    lines.append("[Tier-A]")
    lines.append(_tierA_block(feats))
    lines.append("")
    # Scores block
    lines.append("[Scores]")
    lines.append(_scores_block(scores))

    # Mean & Min（和 print_ui() 里一样）
    vals = [float(scores.get(k, 0.0)) for k in PASS_DIMS]
    mean_score = float(np.mean(vals)) if vals else 0.0
    min_score  = float(min(vals)) if vals else 0.0
    lines.append("")
    lines.append(f"  → Mean: {color_score(mean_score)}")
    lines.append(f"  → Min: {color_score(min_score)}")
    lines.append("-" * 72)

    sym_map = {"PASS": "🟢 PASS", "BORDERLINE": "🟡 BORDERLINE", "FAIL": "🔴 FAIL"}
    sym = sym_map.get(decision, str(decision))
    lines.append(f"Decision: {sym}")

    # 加上 Karen 的大段话
    if response:
        lines.append("")
        lines.append("Presidential Review Board says:")
        lines.append(response.get("karen_text", ""))

    text = "\n".join(lines)

    # ★★★ 关键：去掉所有 ANSI 颜色转义序列 ★★★
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    clean_text = ansi_re.sub('', text)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)



def split_by_tier(feats: Dict[str,float]):
    A = {k: v for k, v in feats.items() if k in TIER_A_KEYS}
    B = {k: v for k, v in feats.items() if k not in TIER_A_KEYS}
    return A, B


def write_dataset_row(csv_path: str, meta: Dict[str, str], scores: Dict[str,float], A: Dict[str,float], B: Dict[str,float]):
    """Append (or create) a CSV row with prefixes: S_*, A_*, B_*.
       If new columns appear later, upgrade the header automatically.
    """
    row = {
        "audio_file": meta.get("audio_file", ""),
        "duration_s": meta.get("duration_s", 0.0),
        "timestamp": meta.get("timestamp", ""),
    }
    for k, v in scores.items():
        row[f"S_{k}"] = float(v)
    for k, v in A.items():
        row[f"A_{k}"] = float(v)
    for k, v in B.items():
        row[f"B_{k}"] = float(v)

    # read existing if present
    rows = []
    existing_header = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            for r in reader:
                rows.append(r)

    # unify header
    header = list(dict.fromkeys((existing_header or []) + list(row.keys())))

    # ensure all rows have all columns
    def fill(r):
        for h in header:
            if h not in r:
                r[h] = ""
        return r
    rows = [fill(r) for r in rows]
    rows.append(fill(row))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

def export_scores_for_td(
        authority_score, trust_score, clarity_score, fluency_score, warmth_score, cadence_score,
        decision, karen_text, feats,
        out_path):

    # 保证 0~1
    clamp = lambda x: float(max(0.0, min(1.0, x)))

    # TD 要的六个参数
    FEATURES_MAP = {
        "INT_DB":   "loudness_db",
        "DUR_PSE":  "pause_mean_s",
        "RATE_SP":  "speaking_rate_wps",
        "F0":       "f0_mean",
        "INT_VAR":  "int_var_std_db",
        "END_CAD":  "final_f0_drop_st"
    }

    features_payload = {
        k: float(feats.get(src, 0.0))
        for k, src in FEATURES_MAP.items()
    }

    payload = {
        "scores": {
            "authority": clamp(authority_score),
            "trust":     clamp(trust_score),
            "clarity":   clamp(clarity_score),
            "fluency":   clamp(fluency_score),
            "warmth":    clamp(warmth_score),
            "cadence":   clamp(cadence_score),
        },
        "decision": str(decision),
        "comment":  str(karen_text),
        "features": features_payload
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def print_ui(scores, feats, decision, response):
        """
        Simple terminal UI output for debugging.
        Mirrors the old behavior (Tier-A block + Scores + Decision + Karen text).
        """

        PASS_DIMS = ["authority","trust","clarity","fluency","warmth","cadence"]

        print("\n" + "-"*72)
        print("[ Presidential Review Board (sim.) ]\n")

        # Tier A block
        print("[Tier-A]")
        print(_tierA_block(feats))
        print("")

        # Scores
        print("[Scores]")
        print(_scores_block(scores))

        # Mean + Min
        vals = [float(scores.get(k, 0.0)) for k in PASS_DIMS]
        mean_score = float(np.mean(vals)) if vals else 0.0
        min_score  = float(min(vals))     if vals else 0.0

        print("")
        print(f"  → Mean: {color_score(mean_score)}")
        print(f"  → Min:  {color_score(min_score)}")
        print("-"*72)

        # Decision
        sym_map = {"PASS":"🟢 PASS", "BORDERLINE":"🟡 BORDERLINE", "FAIL":"🔴 FAIL"}
        print(f"Decision: {sym_map.get(decision, decision)}")

        # Karen Text
        if response:
            print("\nPresidential Review Board says:")
            print(response.get("karen_text",""))
        print("-"*72 + "\n")

def main():
    audio_path = None
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    elif DEFAULT_AUDIO:
        audio_path = DEFAULT_AUDIO

    if not audio_path:
        print("Usage: python karen_score_tiered_v2.py \"C\\path\\to\\audio.wav\"")
        print("Tip: set DEFAULT_AUDIO in the script to avoid passing an argument.")
        sys.exit(1)

    if not os.path.isfile(audio_path):
        print(f"[Error] Audio not found: {audio_path}")
        sys.exit(1)

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    print("[DEBUG] sr loaded:", sr, "  SR const:", SR)
    if len(y)==0:
        print("[Error] Empty audio.")
        sys.exit(1)
    y = librosa.util.normalize(y)

    # --- features ---
    f0_mean, f0_std, jitter = estimate_pitch(y, SR)
    loud_db = rms_db(y)
    _energy = moving_energy(y, int(SR*FRAME_LEN), HOP_LENGTH)  # available if needed
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
    
    # --- extract individual scores for later use ---
    authority_score = scores.get('authority', 0.0)
    trust_score     = scores.get('trust', 0.0)
    clarity_score   = scores.get('clarity', 0.0)
    fluency_score   = scores.get('fluency', 0.0)
    warmth_score    = scores.get('warmth', 0.0)
    cadence_score   = scores.get('cadence', 0.0)

    # --- generate Karen response ---
    responder = KarenResponder(seed=42)

    # 同时提供小写/首字母大写两套 key，避免大小写不一致
    scores_for_responder = {
        **scores,
        **{k.capitalize(): v for k, v in scores.items()}
    }

    response = responder.generate(
    scores=scores, feats=feats, max_tips=3, external_verdict=decision)


    karen_text = response.get("karen_text", "Nice delivery. Keep the cadence and clarity.")

    print_ui(scores, feats, decision, response)


    # assemble payload + save JSON
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
    payload["karen_response"] = response  # 新增


    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # save CSV dataset row
    out_csv = os.path.join(out_dir, "karen_runs.csv")
    meta = {
        "audio_file": os.path.basename(audio_path),
        "duration_s": round(len(y)/SR, 3),
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds')
    }
    write_dataset_row(out_csv, meta, scores, A, B)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")
  # 统一一个给 TD 使用的 Output 目录（项目根目录下的 Output）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    td_out_dir = os.path.join(project_root, "Output")
    os.makedirs(td_out_dir, exist_ok=True)

    export_scores_for_td(
        authority_score, trust_score, clarity_score, fluency_score, warmth_score, cadence_score,
        decision, karen_text,
        feats,
        out_path=os.path.join(td_out_dir, "td_scores.json")
    )
    
    save_full_report_text(
        scores, feats, decision, response,
        out_path=os.path.join(td_out_dir, "karen_report.txt")
    )







if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
