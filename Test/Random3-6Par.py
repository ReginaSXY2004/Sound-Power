import random, os
import numpy as np
import librosa, soundfile as sf
import pyrubberband as pyrb

# ---------- 路径 ----------
input_file = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\TestAudioOutput\2_abstract.wav"
output_dir = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\修改完参数Output"
os.makedirs(output_dir, exist_ok=True)

# ---------- 载入 ----------
y, sr = librosa.load(input_file, sr=None)
y = librosa.util.fix_length(y, size=len(y))  # ✅ 正确写法


# ---------- 功能块（严格按表格） ----------
def apply_gain_db(y, gain_db):
    factor = 10 ** (gain_db / 20.0)
    return np.clip(y * factor, -1.0, 1.0)

def INT_DB(y):
    # 只增不减：+3 ~ +6 dB
    g = random.uniform(3.0, 6.0)
    print(f"[INT-DB] +{g:.2f} dB (toward 70–75 dB SPL)")
    return apply_gain_db(y, g)

def DUR_PSE(y, sr):
    # 前置 0.6–1.2 s 静音
    pause_len = int(random.uniform(0.6, 1.2) * sr)
    print(f"[DUR-PSE] prepend silence {pause_len/sr:.2f}s")
    return np.concatenate([np.zeros(pause_len, dtype=y.dtype), y])

def RATE_SP(y, sr):
    # 目标 3–4 音节/秒：这里采用保守整体放慢 0.75~0.90
    # （没有分词信息时，用 time-stretch 近似）
    rate = random.uniform(0.75, 0.90)   # <1 变慢
    print(f"[RATE-SP] time-stretch factor={rate:.2f}  (slower to ~3–4 syl/s)")
    try:
        return pyrb.time_stretch(y, sr, rate)
    except Exception as e:
        print("  rubberband unavailable, fallback to librosa:", e)
        return librosa.effects.time_stretch(y, rate)

def F0_LOWER(y, sr):
    # 男/女目标区间→做“降低基频”的相对策略：-1.5 ~ -3.5 半音
    n_steps = random.uniform(-3.5, -1.5)
    print(f"[F0] pitch shift {n_steps:.2f} semitones (down)")
    try:
        return pyrb.pitch_shift(y, sr, n_steps)
    except Exception as e:
        print("  rubberband unavailable, fallback to librosa:", e)
        return librosa.effects.pitch_shift(y, sr, n_steps)

def INT_VAR(y, sr):
    # 在全段上做缓慢响度起伏：±(6~10)dB，使用低频正弦/LFO 包络
    peak_db = random.uniform(6.0, 10.0)
    lfo_hz   = random.uniform(0.2, 0.5)     # 缓慢起伏
    print(f"[INT-VAR] ±{peak_db:.1f} dB with LFO {lfo_hz:.2f} Hz")
    t = np.arange(len(y))/sr
    # 正弦范围 [-1,1] → dB 起伏 [-peak, +peak]
    mod_db = peak_db * np.sin(2*np.pi*lfo_hz*t)
    mod_lin = 10 ** (mod_db/20.0)
    y_mod = np.clip(y * mod_lin, -1.0, 1.0)
    return y_mod.astype(y.dtype)

def END_CAD(y, sr):
    # 最后 15–20% 段做“下滑音高”：-3.0 ~ -4.0 半音（约 15–20% 频降）
    tail_ratio = random.uniform(0.15, 0.20)
    n_tail = int(len(y) * tail_ratio)
    print(f"[END-CAD] last {tail_ratio*100:.1f}% pitch drop ~ −3~−4 semitones")

    head = y[:-n_tail] if n_tail>0 else y
    tail = y[-n_tail:] if n_tail>0 else np.array([], dtype=y.dtype)

    if n_tail > 0:
        # 简化：把尾段整体下移固定半音值（近似“降调收尾”）
        n_steps = random.uniform(-4.0, -3.0)
        try:
            tail_shift = pyrb.pitch_shift(tail, sr, n_steps)
        except Exception as e:
            print("  rubberband unavailable, fallback to librosa:", e)
            tail_shift = librosa.effects.pitch_shift(tail, sr, n_steps)
        # 轻微减 2 dB 做“收尾”
        tail_shift = apply_gain_db(tail_shift, -2.0)
        y_out = np.concatenate([head, tail_shift])
    else:
        y_out = y
    return y_out

# ---------- 随机选 3–6 个参数并应用 ----------
ops_map = {
    "INT-DB": INT_DB,
    "DUR-PSE": lambda y: DUR_PSE(y, sr),
    "RATE-SP": lambda y: RATE_SP(y, sr),
    "F0":      lambda y: F0_LOWER(y, sr),
    "INT-VAR": lambda y: INT_VAR(y, sr),
    "END-CAD": lambda y: END_CAD(y, sr),
}
all_keys = list(ops_map.keys())
chosen = random.sample(all_keys, random.randint(3, 6))
print("\nApplying (from spec):", chosen, "\n")

y_mod = y.copy()
for k in chosen:
    y_mod = ops_map[k](y_mod)

# ---------- 保存 ----------
out_path = os.path.join(output_dir, f"2_randomized_spec_{random.randint(1000,9999)}.wav")
sf.write(out_path, y_mod, sr)
print("\n✅ Saved:", out_path)
