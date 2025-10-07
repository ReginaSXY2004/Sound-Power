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
DEFAULT_AUDIO = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\TestAudioInput\1.m4a"

SR = 16000               # 统一采样率
FRAME_HOP = 0.02         # 20ms hop
FRAME_LEN = 0.04         # 40ms window
HOP_LENGTH = int(SR * FRAME_HOP)
N_FFT = 2048

# Karen 的“社会偏见”权重（可迭代调参）
# 线性打分后走 sigmoid → 0~1
WEIGHTS = {
    "authority": {
        "f0_mean": -0.45,
        "f0_std": -0.10,
        "loudness_db": 0.35,
        "pause_ratio": -0.10,          # 原 -0.30 → 降低惩罚，交给细分特征
        "pause_long_ratio": 0.28,      # 长停顿加分（节奏自信/权威）
        "filler_like_ratio": -0.35,    # 填充音减分（显得不稳）
        "jitter": -0.20,
    },
    "trust": {
        "jitter": -0.28,               # 略放松
        "shimmer": -0.20,              # 略放松
        "clarity": 0.30,
        "speaking_rate_wps": 0.12,
        "noise_floor_db": -0.08,
        "filler_like_ratio": -0.30,    # 填充音损伤“可信”
        "pause_cv": -0.08              # 节奏过度忽快忽慢会降低可信
    },
    "clarity": {
        "clarity": 0.45,
        "spectral_rolloff": -0.05,
        "spectral_centroid": -0.05,
        "pause_micro_ratio": -0.08,    # 微停顿太多可能割裂清晰度
        "pause_ratio": -0.06
    },
    "fluency": {
        "pause_ratio": -0.18,          # 总停顿仍减分，但比原来温和
        "speaking_rate_wps": 0.22,
        "f0_std": -0.05,
        "filler_like_ratio": -0.40,    # 流畅度里最重惩罚填充音
        "pause_long_ratio": 0.10       # 恰当长停顿可视为有节奏的“呼吸”
    },
    "warmth": {     # 亲和：中低音域、响度不过度、抖动不过大
        "f0_mean": -0.12,
        "loudness_db": 0.10,
        "jitter": -0.08,
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
    # 负向触发（阴阳怪气）
    (lambda f,s: f["f0_mean"] > 200 and s["authority"] < 0.55, "High pitch gives me intern vibes."),
    (lambda f,s: f["filler_like_ratio"] > 0.04, "The little 'uh' and 'um'—like pebbles in your shoes."),
    (lambda f,s: f["pause_micro_ratio"] > 0.08 and s["fluency"] < 0.5, "Choppy rhythm. This isn’t a buffering icon."),
    (lambda f,s: f["loudness_db"] < -18, "Speak up. This is a rally, not a library."),
    (lambda f,s: f["jitter"] > 0.03, "Your voice shakes like my iced latte."),
    (lambda f,s: s["trust"] < 0.4 and s["clarity"] < 0.5, "Hard to trust what I can barely parse."),
    # 正向触发（真鼓励）
    (lambda f,s: f["pause_long_ratio"] > 0.06 and s["authority"] > 0.6, "Those deliberate pauses? Presidential."),
    (lambda f,s: s["clarity"] > 0.65 and s["fluency"] > 0.6, "Clean articulation and steady flow—your message lands."),
    (lambda f,s: s["warmth"] > 0.6 and s["trust"] > 0.55, "You sound like someone people want to follow, not just hear."),
    (lambda f,s: s["authority"] > 0.7, "Commanding presence. The room moves when you do.")
]


# ------------ 工具函数 ------------
def contiguous_runs(mask):
    """返回 (start_idx, end_idx) 的连续区间，end为包含式"""
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

def estimate_pause_features(y, sr):
    """
    返回：
      pause_ratio         - 静音占比（总停顿时间 / 总时长）
      pause_long_ratio    - 长停顿（≥350ms）占比
      pause_micro_ratio   - 微停顿（50–250ms）占比
      filler_like_ratio   - 疑似填充音占比（短、低能量、voiced、f0稳定）
      pause_cv            - 停顿时长变异系数（std/mean）
    """
    # ---- 帧化 & 能量 ----
    frame_len = int(sr * FRAME_LEN)      # e.g., 0.04s
    hop_len   = HOP_LENGTH               # e.g., 0.02s
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    energy = np.mean(frames**2, axis=0)
    med = np.median(energy) + 1e-9
    thr = med * 0.25
    is_sil = energy < thr  # 静音掩码（基于相对阈值）

    # ---- voicing（用 pyin voiced_prob 近似）----
    # 用更长一点的窗避免“两周期不足”的警告
    pyin_frame_len = max(frame_len, int(sr * 0.06))
    f0, _, vprob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=pyin_frame_len, hop_length=hop_len
    )
    vprob = np.nan_to_num(vprob, nan=0.0)
    is_voiced = vprob > 0.6

    # f0 稳定性（供填充音识别）
    f0_s = np.nan_to_num(f0, nan=0.0)
    f0_diff = np.abs(np.diff(f0_s))
    f0_diff = np.concatenate([[0.0], f0_diff])  # 对齐长度
    f0_stable = f0_diff < 3.5  # Hz 阈值

    # ---- 长度对齐（关键修复点）----
    n = min(len(is_sil), len(is_voiced), len(energy), len(f0_stable))
    is_sil     = is_sil[:n]
    is_voiced  = is_voiced[:n]
    f0_stable  = f0_stable[:n]
    energy     = energy[:n]
    low_energy = (energy < (med * 0.6))

    # ---- 统计连续静音段（得到 pause_durs）----
    runs = contiguous_runs(is_sil)  # [(start, end), ...], end 包含
    ms_per_hop = (hop_len / sr) * 1000.0

    def dur_ms(a, b):
        # 区间长度 = 帧数 * hop 时间；包含式区间 → 帧数 = (b - a + 1)
        return (b - a + 1) * ms_per_hop

    pause_durs = [dur_ms(a, b) for (a, b) in runs]  # ←← 这里就定义了
    total_time_s = n * (hop_len / sr) + 1e-9
    pause_time_s = sum(pause_durs) / 1000.0

    # ---- 分箱：long / micro ----
    long_thr = 350.0   # ms
    micro_lo, micro_hi = 50.0, 250.0

    long_time_s  = sum(d for d in pause_durs if d >= long_thr) / 1000.0
    micro_time_s = sum(d for d in pause_durs if micro_lo <= d <= micro_hi) / 1000.0

    # ---- 疑似填充音：非静音 & Voiced & 低能量 & f0稳定，时长 80–300ms ----
    filler_mask = (~is_sil) & is_voiced & low_energy & f0_stable
    filler_runs = contiguous_runs(filler_mask)
    filler_time_s = 0.0
    for (a, b) in filler_runs:
        d = dur_ms(a, b)
        if 80.0 <= d <= 300.0:
            filler_time_s += d / 1000.0

    # ---- 比例 & 变异系数 ----
    pause_ratio        = float(pause_time_s  / total_time_s)
    pause_long_ratio   = float(long_time_s   / total_time_s)
    pause_micro_ratio  = float(micro_time_s  / total_time_s)
    filler_like_ratio  = float(filler_time_s / total_time_s)

    if len(pause_durs) >= 2 and np.mean(pause_durs) > 1e-9:
        pause_cv = float(np.std(pause_durs) / (np.mean(pause_durs) + 1e-9))
    else:
        pause_cv = 0.0

    return {
        "pause_ratio": pause_ratio,
        "pause_long_ratio": pause_long_ratio,
        "pause_micro_ratio": pause_micro_ratio,
        "filler_like_ratio": filler_like_ratio,
        "pause_cv": pause_cv
    }



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
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=int(sr * 0.06),   # ← 用更长一点的帧长
        hop_length=HOP_LENGTH
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
    frame_len = int(sr*FRAME_LEN)
    hop_len = HOP_LENGTH
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    # 能量幅度（避免0）
    amps = np.sqrt(np.mean(frames**2, axis=0)) + 1e-6
    da = np.abs(np.diff(amps)) / np.maximum(amps[:-1], 1e-6)
    # 用中位数抗极端值，并剪裁到合理范围
    sh = float(np.median(da))
    return float(np.clip(sh, 0.0, 0.3))


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
    pos, neg = [], []
    for cond, text in KAREN_SNARKS:
        try:
            if cond(feats, dim_scores):
                # 简单规则：含有“Presidential/clean/want to follow/Commanding”关键词视作正向
                if any(k in text for k in ["Presidential", "Clean", "follow", "Commanding", "lands"]):
                    pos.append(text)
                else:
                    neg.append(text)
        except Exception:
            continue

    # 规则：最多展示3条；优先展示1-2条负向 + 1-2条正向，避免刷屏
    lines = []
    if dim_scores["authority"] >= 0.6 or dim_scores["trust"] >= 0.6:
        # 高分：正面为主，最多1条温和建议
        lines.extend(pos[:2] or ["Strong stance."])
        if neg:
            lines.append("Refine one detail: " + neg[0])
    else:
        # 低分：负面为主，但保留一条'假鼓励'
        lines.extend(neg[:2] or ["That’s cute, but we’re looking for someone with more… authority."])
        if pos:
            lines.append("Hidden potential: " + pos[0])

    # 加收束语（竞选叙事）
    if all(dim_scores[k] >= v for k, v in PASS_RULE.items()):
        lines.append("Campaign verdict: APPROVED. See you on the ballot.")
    else:
        lines.append("Campaign verdict: REJECTED. Rally your voice and come back.")

    return " ".join(lines[:4])


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
    pause_feats = estimate_pause_features(y, SR)
    pause_ratio = pause_feats["pause_ratio"]

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
        "clarity": clarity,
        "pause_long_ratio": pause_feats["pause_long_ratio"],
        "pause_micro_ratio": pause_feats["pause_micro_ratio"],
        "filler_like_ratio": pause_feats["filler_like_ratio"],
        "pause_cv": pause_feats["pause_cv"],

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

    # ✅ 新增：numpy→Python类型转换
    def _to_py(o):
        import numpy as np
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    # 转换后写出
    out_payload = _to_py(asdict(card))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"Profile Card saved to: {out_json}")


if __name__ == "__main__":
    main()
