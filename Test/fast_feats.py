# -*- coding: utf-8 -*-
"""
fast_feats.py  
提取修改后的 6 个参数，用于 TouchDesigner。

输出 JSON 字段（全部 float）：
{
  "INT_DB":  x,
  "DUR_PSE": x,
  "RATE_SP": x,
  "F0":      x,
  "INT_VAR": x,
  "END_CAD": x
}
"""

import sys
import json
import numpy as np
import librosa

# -------------------------
# 1. Loudness (INT_DB)
# -------------------------
def estimate_int_db(y):
    """整体响度：简单用 RMS -> dB"""
    if len(y) == 0:
        return 0.0
    rms = np.sqrt(np.mean(y ** 2))
    db = 20 * np.log10(rms + 1e-7)
    return float(db)


# -------------------------
# 2. Pause (DUR_PSE)
# -------------------------
def estimate_pause(y, sr):
    """简单估计停顿：寻找 < -40dB 的区域占比"""
    if len(y) == 0:
        return 0.0

    frame = librosa.util.frame(y, frame_length=2048, hop_length=512).astype(np.float32)
    rms = np.sqrt(np.mean(frame**2, axis=0)) + 1e-7
    db = 20*np.log10(rms)

    pauses = np.mean(db < -40)   # 百分比
    return float(pauses)


# -------------------------
# 3. Rate (RATE_SP)
# -------------------------
def estimate_rate(y, sr):
    """基于 onset 估计语速"""
    if len(y) == 0:
        return 0.0
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # 新版 librosa 必须使用关键字参数
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=3,
        delta=0.5,
        wait=5
    )

    dur = len(y) / sr
    if dur <= 0:
        return 0.0
    
    return float(len(peaks) / dur)



# -------------------------
# 4. Pitch (F0)
# -------------------------
def estimate_pitch(y, sr):
    """平均 F0"""
    if len(y) == 0:
        return 0.0

    try:
        f0, _, _ = librosa.pyin(
            y, 
            fmin=65, fmax=400,
            frame_length=1024,
            sr=sr
        )
        f0 = f0[np.isfinite(f0)]
        if len(f0)==0:
            return 0.0
        return float(np.mean(f0))
    except:
        return 0.0


# -------------------------
# 5. INT_VAR
# -------------------------
def estimate_int_var(y, sr):
    """响度方差：RMS 的方差"""
    if len(y)==0:
        return 0.0

    # 分帧
    hop = 512
    frame = librosa.util.frame(y, frame_length=2048, hop_length=hop)
    rms = np.sqrt(np.mean(frame**2, axis=0)) + 1e-7
    db = 20*np.log10(rms)
    return float(np.std(db))


# -------------------------
# 6. END_CAD
# -------------------------
def estimate_end_cadence(y, sr):
    """最后 20% 段的 pitch 下降量（半音）"""
    if len(y)==0:
        return 0.0

    n = len(y)
    tail = y[int(n*0.8):]

    try:
        f0, _, _ = librosa.pyin(
            tail,
            fmin=65,
            fmax=400,
            frame_length=1024,
            sr=sr
        )
        f0 = f0[np.isfinite(f0)]
        if len(f0) < 2:
            return 0.0

        start = np.mean(f0[:len(f0)//3])
        end   = np.mean(f0[-len(f0)//3:])
        if start <= 0 or end <= 0:
            return 0.0

        semitone = 12*np.log2(end/start)
        return float(semitone)
    except:
        return 0.0


# ===============================================================
#                主执行：提取六参数并写入 JSON
# ===============================================================
def extract_six(infile, outfile):
    y, sr = librosa.load(infile, sr=None)

    feats = {
        "INT_DB":   estimate_int_db(y),
        "DUR_PSE":  estimate_pause(y, sr),
        "RATE_SP":  estimate_rate(y, sr),
        "F0":       estimate_pitch(y, sr),
        "INT_VAR":  estimate_int_var(y, sr),
        "END_CAD":  estimate_end_cadence(y, sr),
    }

    # 统一转 float，避免 numpy.float32 写 JSON 报错
    feats = {k: float(v) for k, v in feats.items()}

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)

    print(f"[FastFeat] Saved: {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fast_feats.py input.wav output.json")
        sys.exit(1)

    extract_six(sys.argv[1], sys.argv[2])
