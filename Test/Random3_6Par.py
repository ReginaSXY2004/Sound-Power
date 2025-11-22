# -*- coding: utf-8 -*-
"""
Random3_6Par_final.py
用于 OneClickFix：
  1. 读取 2_abstract.wav
  2. 随机修改 3–6 个参数（保持你原来的算法）
  3. 保存修改后的 wav
  4. 用 FastFeat 获取六参数 → six.json
  5. 调用 karen_score_tiered_v2 生成 JSON & CSV
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
import subprocess
import json
import pyrubberband as pyrb

# ============================================================
# 路径配置（你原来的固定路径）
# ============================================================

PROJECT_ROOT = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power"
ABSTRACT_WAV = os.path.join(PROJECT_ROOT, "TestAudioOutput", "2_abstract.wav")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "修改完参数Output")
FASTFEAT_PY  = os.path.join(PROJECT_ROOT, "Test", "fast_feats.py")
KAREN_PY     = os.path.join(PROJECT_ROOT, "Test", "karen_score_tiered_v2.py")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 载入音频
# ============================================================
y, sr = librosa.load(ABSTRACT_WAV, sr=None)
y = librosa.util.fix_length(y, size=len(y))

# ============================================================
# 你的原始算法完全保留
# ============================================================

def apply_gain_db(y, gain_db):
    factor = 10 ** (gain_db / 20.0)
    return np.clip(y * factor, -1.0, 1.0)

def INT_DB(y):
    g = random.uniform(3.0, 6.0)
    print(f"[INT-DB] +{g:.2f} dB")
    return apply_gain_db(y, g)

def DUR_PSE(y, sr):
    pause_len = int(random.uniform(0.6, 1.2) * sr)
    print(f"[DUR-PSE] prepend {pause_len/sr:.2f}s silence")
    return np.concatenate([np.zeros(pause_len, dtype=y.dtype), y])

def RATE_SP(y, sr):
    rate = random.uniform(0.75, 0.90)
    print(f"[RATE-SP] stretch={rate:.2f}")
    try:
        return pyrb.time_stretch(y, sr, rate)
    except:
         return librosa.effects.time_stretch(y, rate=rate)


def F0_LOWER(y, sr):
    n_steps = random.uniform(-3.5, -1.5)
    print(f"[F0] {n_steps:.2f} semitone down")
    try:
        return pyrb.pitch_shift(y, sr, n_steps)
    except:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def INT_VAR(y, sr):
    peak_db = random.uniform(6.0, 10.0)
    lfo_hz  = random.uniform(0.2, 0.5)
    print(f"[INT-VAR] ±{peak_db:.1f} dB LFO {lfo_hz:.2f} Hz")

    t = np.arange(len(y))/sr
    mod_db = peak_db * np.sin(2*np.pi*lfo_hz*t)
    mod_lin = 10 ** (mod_db/20.0)
    return np.clip(y * mod_lin, -1.0, 1.0)

def END_CAD(y, sr):
    tail_ratio = random.uniform(0.15, 0.20)
    n_tail = int(len(y) * tail_ratio)
    print(f"[END-CAD] last {tail_ratio*100:.1f}% drop")

    head = y[:-n_tail]
    tail = y[-n_tail:]

    n_steps = random.uniform(-4.0, -3.0)
    try:
        tail_shift = pyrb.pitch_shift(tail, sr, n_steps)
    except:
        tail_shift = librosa.effects.pitch_shift(tail, sr=sr, n_steps=n_steps)


    tail_shift = apply_gain_db(tail_shift, -2.0)
    return np.concatenate([head, tail_shift])

# ============================================================
# 随机选择 3–6 个操作 + 应用
# ============================================================

ops_map = {
    "INT-DB": INT_DB,
    "DUR-PSE": lambda y: DUR_PSE(y, sr),
    "RATE-SP": lambda y: RATE_SP(y, sr),
    "F0": lambda y: F0_LOWER(y, sr),
    "INT-VAR": lambda y: INT_VAR(y, sr),
    "END-CAD": lambda y: END_CAD(y, sr),
}

chosen = random.sample(list(ops_map.keys()), random.randint(3, 6))
print("\nChosen ops:", chosen, "\n")

y_mod = y.copy()
for k in chosen:
    y_mod = ops_map[k](y_mod)

# ============================================================
# 保存修改后的结果
# ============================================================

rand_name = f"2_randomized_spec_{random.randint(1000,9999)}.wav"
modified_wav = os.path.join(OUTPUT_DIR, rand_name)

sf.write(modified_wav, y_mod, sr)
print("Saved modified wav:", modified_wav)

# ============================================================
# 生成 six.json（由 FastFeat 计算六个参数）
# ============================================================

six_json_path = os.path.join(OUTPUT_DIR, "six.json")

print("\n[FastFeat] Running fast_feats.py to compute 6 params...\n")
subprocess.check_call([
    "python", FASTFEAT_PY, modified_wav, six_json_path
])

print("Generated Six JSON:", six_json_path)

# ============================================================
# 调用 karen_score_tiered_v2（保持你原来的流程）
# ============================================================

print("\n[Karen] Running karen_score_tiered_v2.py...\n")
subprocess.call([
    "python", KAREN_PY, modified_wav
])

print("\nAll Done.\n")
