# karen_score.py
# Usage:
#   python karen_score.py
#   python karen_score.py "C:\path\to\your\audio.m4a"

import os, sys, json, time, math
from dataclasses import dataclass, asdict

import numpy as np
import librosa
import librosa.display

# ------------ 配置区（你可以随时改） ------------
DEFAULT_AUDIO = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\TestAudioInput\Obama.wav"

SR = 16000               # 统一采样率
FRAME_HOP = 0.02         # 20ms hop
FRAME_LEN = 0.04         # 40ms window
HOP_LENGTH = int(SR * FRAME_HOP)
N_FFT = 2048

# Karen 的“社会偏见”权重（可迭代调参）
# 线性打分后走 sigmoid → 0~1
WEIGHTS = {
    "authority": {  # 权威：低音高/高响度/低停顿/更稳定
        "f0_mean": -0.45,
        "f0_std": -0.10,
        "loudness_db": 0.35,
        "pause_ratio": -0.30,
        "jitter": -0.20,
    },
    "trust": {      # 信任：稳定、清晰、噪声低、语速适中
        "jitter": -0.30,
        "shimmer": -0.25,
        "clarity": 0.30,
        "speaking_rate_wps": 0.15,
        "noise_floor_db": -0.10,  # 噪声越高，信任越低
    },
    "clarity": {    # 清晰：高能量瞬变+谱重心合理+rolloff适中
        "clarity": 0.45,
        "spectral_rolloff": -0.05,
        "spectral_centroid": -0.05,
        "pause_ratio": -0.10,
    },
    "fluency": {    # 流畅：少停顿、语速合适、韵律稳定
        "pause_ratio": -0.40,
        "speaking_rate_wps": 0.25,
        "f0_std": -0.05,
    },
    "warmth": {     # 亲和：中低音域、响度不过度、抖动不过大
        "f0_mean": -0.15,
        "loudness_db": 0.10,
        "jitter": -0.10,
        "spectral_centroid": -0.10,
    }
}

BIASES = {  # 每个维度的偏置
    "authority": 0.0,
    "trust": 0.0,
    "clarity": 0.0,
    "fluency": 0.0,
    "warmth": 0.0
}

PASS_RULE = {  # 通过条件（可改）
    "authority": 0.55,
    "trust": 0.50,
    "clarity": 0.50,
    "fluency": 0.50,
}

# Karen 的刻薄语句触发规则（简单模板）
KAREN_SNARKS = [
    # (条件, 文案)
    (lambda f,s: f["f0_mean"] > 200 and s["authority"] < 0.5, "High pitch gives me intern vibes."),
    (lambda f,s: f["pause_ratio"] > 0.20, "Too many pauses—your confidence leaks between silences."),
    (lambda f,s: f["loudness_db"] < -18, "Speak up. We can't hire whispers."),
    (lambda f,s: f["jitter"] > 0.03, "Your voice shakes like my iced latte."),
    (lambda f,s: f["clarity"] < 0.45, "Muddy articulation. Try separating your words from each other."),
    (lambda f,s: f["speaking_rate_wps"] < 1.6, "Pick up the pace; we’re not at bedtime stories."),
    (lambda f,s: f["spectral_centroid"] > 3500, "A little less sizzle, a little more substance."),
]

# ------------ 工具函数 ------------
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def rms_db(y):
    rms = np.sqrt(np.mean(np.square(y))) + 1e-9
    return 20 * np.log10(rms + 1e-12)

def moving_energy(y, frame_len, hop_len):
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    return np.mean(frames**2, axis=0)

def estimate_pauses(energy, thresh_ratio=0.25):
    # 使用能量中位数的比例阈值估计静音/停顿
    med = np.median(energy) + 1e-9
    thr = med * thresh_ratio
    pauses = (energy < thr).astype(np.float32)
    return float(np.mean(pauses))

def estimate_speaking_rate(y, sr):
    # 粗略：过零率+能量峰近似节律 -> 估算每秒“词”（非常粗糙，但可用来驱动分数）
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(sr*FRAME_LEN), hop_length=int(sr*FRAME_HOP))[0]
    # 峰值检测（节律 proxy）
    peaks = (zcr[1:-1] > zcr[:-2]) & (zcr[1:-1] > zcr[2:]) & (zcr[1:-1] > (np.mean(zcr) + np.std(zcr)))
    approx_units_per_sec = peaks.sum() / (len(zcr)/ (sr*FRAME_HOP))
    # 映射到 1.0~3.5 之间
    return float(np.clip(approx_units_per_sec, 0.8, 4.0))

def spectral_features(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean()
    return float(centroid), float(rolloff)

def estimate_pitch(y, sr):
    # 使用 PYIN 提取基频（NaN 会被忽略）
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=int(sr*FRAME_LEN), hop_length=HOP_LENGTH
    )
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 0.0, 0.0, 0.0
    f0_mean = float(np.mean(f0))
    f0_std = float(np.std(f0))
    # 近似 jitter: 相邻帧频率变化的相对抖动
    df = np.abs(np.diff(f0))
    jitter = float(np.mean(df / (f0[:-1] + 1e-9))) if len(df) > 0 else 0.0
    return f0_mean, f0_std, jitter

def estimate_shimmer(y, sr):
    # 近似 shimmer：短帧能量的相对波动
    frame_len = int(sr*FRAME_LEN)
    hop_len = HOP_LENGTH
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    amps = np.sqrt(np.mean(frames**2, axis=0)) + 1e-9
    da = np.abs(np.diff(amps)) / (amps[:-1] + 1e-9)
    return float(np.mean(da)) if len(da) > 0 else 0.0

def estimate_noise_floor_db(y, sr):
    # 简单估计：取能量最低的 10% 帧的均值作为“底噪”
    frame_len = int(sr*FRAME_LEN)
    hop_len = HOP_LENGTH
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    e = np.mean(frames**2, axis=0)
    k = max(1, int(0.1 * len(e)))
    floor = np.mean(np.sort(e)[:k]) + 1e-12
    return 10 * np.log10(floor)

def estimate_clarity(y, sr):
    # 语音清晰度粗估：过零率偏低 + 高频不过度 + 能量集中
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(sr*FRAME_LEN), hop_length=HOP_LENGTH)[0]
    centroid, rolloff = spectral_features(y, sr)
    # 归一：给出 0~1 的清晰度（经验式）
    zcr_norm = 1.0 - np.clip((zcr.mean() - 0.05) / 0.15, 0, 1)          # ZCR越高，越不清晰 → 取反
    cent_norm = 1.0 - np.clip((centroid - 2500) / 2500, 0, 1)           # 过高的谱重心→刺耳
    roll_norm = 1.0 - np.clip((rolloff - 6000) / 6000, 0, 1)            # 过多高频尾巴→浑浊
    clarity = 0.5*zcr_norm + 0.25*cent_norm + 0.25*roll_norm
    return float(np.clip(clarity, 0, 1))

def standardize_feature(name, val):
    # 为了让权重更稳，做个粗糙标准化（经验中心+尺度）
    anchors = {
        "f0_mean": (180, 60),               # Hz
        "f0_std": (35, 20),                 # Hz
        "jitter": (0.02, 0.02),             # ratio
        "shimmer": (0.08, 0.05),            # ratio
        "loudness_db": (-15, 6),            # dBFS近似
        "pause_ratio": (0.15, 0.15),        # 0~1
        "speaking_rate_wps": (2.2, 0.7),    # words/sec 近似
        "spectral_centroid": (2500, 1200),  # Hz
        "spectral_rolloff": (6000, 2500),   # Hz
        "noise_floor_db": (-40, 8),         # dB
        "clarity": (0.6, 0.2),              # 0~1
    }
    mu, sd = anchors.get(name, (0.0, 1.0))
    return (val - mu) / (sd + 1e-9)

def score_dimensions(feats):
    # feats: 原始物理量; 先标准化再线性加权 → sigmoid
    xz = {k: standardize_feature(k, v) for k, v in feats.items()}
    dim_scores = {}
    for dim, wmap in WEIGHTS.items():
        z = BIASES.get(dim, 0.0)
        for fname, w in wmap.items():
            z += w * xz.get(fname, 0.0)
        dim_scores[dim] = float(sigmoid(z))
    return dim_scores

def make_karen_comment(feats, dim_scores):
    lines = []
    # 命中规则的刻薄话
    for cond, text in KAREN_SNARKS:
        try:
            if cond(feats, dim_scores):
                lines.append(text)
        except Exception:
            continue
    # 兜底
    if not lines:
        if dim_scores["authority"] < 0.5:
            lines.append("That’s cute, but we’re looking for someone with more… authority.")
        else:
            lines.append("Not bad. Try sounding like you mean it.")
    # 加一条“假鼓励”
    if dim_scores["trust"] < 0.5:
        lines.append("I believe in growth… for other candidates. You’ll get there.")
    else:
        lines.append("See? A little effort goes a long way.")
    return " ".join(lines)

def pass_fail(dim_scores):
    ok = all(dim_scores[k] >= v for k, v in PASS_RULE.items())
    return "PASS" if ok else "FAIL"

@dataclass
class ProfileCard:
    candidate_id: str
    predicted_role: str
    voice_bias: str
    decision: str
    scores: dict
    features: dict

def build_profile_card(feats, scores, decision):
    # 粗略生成“偏好角色”与“Voice Bias”标签
    role = "Assistant Personality Type"
    if scores["authority"] > 0.65 and scores["trust"] > 0.55:
        role = "Team Lead Personality Type"
    if scores["fluency"] > 0.7 and scores["clarity"] > 0.65:
        role = "Spokesperson Personality Type"

    # Voice Bias 标签（示例）
    bias_tags = []
    bias_tags.append("feminine" if feats["f0_mean"] > 200 else "masculine-ish")
    bias_tags.append("East-Coast Accent? (proxy)")  # 这里只做占位符，未来可用更复杂方法估算
    bias_tags.append("polite under stress" if feats["pause_ratio"] > 0.2 else "decisive pacing")

    return ProfileCard(
        candidate_id=f"#{np.random.randint(10000, 99999)}",
        predicted_role=role,
        voice_bias=" / ".join(bias_tags),
        decision=decision,
        scores=scores,
        features=feats
    )

def print_ui(scores, feats, decision, karen_text):
    # 伪装成“企业面试 AI”界面输出
    conf = int(round(100 * ((scores["authority"] + scores["trust"]) / 2)))
    accent_dev = abs(standardize_feature("spectral_centroid", feats["spectral_centroid"]))
    trust_index = 10 * scores["trust"]

    print("\n[ Calibrating Karen… ]")
    for _ in range(10):
        time.sleep(0.05)
        print("▮", end="", flush=True)
    print("\n")

    print(f"Confidence Analysis {conf}% | Accent Deviation {accent_dev:.2f} | Trust Index {trust_index:.1f}/10")
    print("-" * 72)
    for k in ["authority", "trust", "clarity", "fluency", "warmth"]:
        print(f"{k.capitalize():<10}: {scores[k]:.2f}")
    print("-" * 72)
    lamp = "🟢 PASS" if decision == "PASS" else "🔴 FAIL"
    print(f"Decision: {lamp}")
    print("\nKaren says:")
    print(karen_text)
    print("-" * 72)

def main():
    audio_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_AUDIO
    if not os.path.isfile(audio_path):
        print(f"[Error] Audio not found: {audio_path}")
        sys.exit(1)

    # 读取音频
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    y = librosa.util.normalize(y)

    # 特征提取
    f0_mean, f0_std, jitter = estimate_pitch(y, SR)
    loud_db = rms_db(y)
    energy = moving_energy(y, int(SR*FRAME_LEN), HOP_LENGTH)
    pause_ratio = estimate_pauses(energy, thresh_ratio=0.25)
    speaking_rate = estimate_speaking_rate(y, SR)
    centroid, rolloff = spectral_features(y, SR)
    shimmer = estimate_shimmer(y, SR)
    clarity = estimate_clarity(y, SR)
    noise_db = estimate_noise_floor_db(y, SR)

    feats = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "loudness_db": loud_db,
        "pause_ratio": pause_ratio,
        "speaking_rate_wps": speaking_rate,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "noise_floor_db": noise_db,
        "clarity": clarity
    }

    # 打分
    scores = score_dimensions(feats)
    decision = pass_fail(scores)
    karen_text = make_karen_comment(feats, scores)
    print_ui(scores, feats, decision, karen_text)

    # 生成 Profile Card 并保存
    card = build_profile_card(feats, scores, decision)
    out_dir = os.path.join(os.path.dirname(audio_path), "Output")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "karen_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(card), f, ensure_ascii=False, indent=2)
    print(f"Profile Card saved to: {out_json}")

if __name__ == "__main__":
    main()
