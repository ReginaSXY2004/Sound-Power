# ethereal_blur_drum_fixed.py — voice→drum morph, de‑machine‑gun, bug‑fixed
# 用法：
#   python ethereal_blur_drum_fixed.py "TestAudioInput/1.m4a" "TestAudioOutput/1_abstract.wav"

import sys, math
import numpy as np
import librosa, soundfile as sf
from scipy import signal
from scipy.signal import iirpeak, lfilter

# ============== 基础工具 ==============
def normalize(x, peak=0.95):
    x = x / (np.max(np.abs(x)) + 1e-12)
    return x * peak

def lowpass(x, sr, cutoff=6500, order=4):
    ny = 0.5 * sr
    b, a = signal.butter(order, cutoff/ny, btype='low')
    return signal.lfilter(b, a, x)

def make_ir(sr, seconds=4.0, decay=2.3, lp=4000):
    """更暗一点的合成 IR，减少嘶声/颗粒被放大"""
    n = int(seconds*sr)
    t = np.linspace(0, seconds, n)
    env = np.exp(-decay*t)
    ir  = np.random.randn(n) * env
    ny = 0.5*sr
    b,a = signal.butter(6, lp/ny, btype='low')
    ir  = signal.lfilter(b,a,ir)
    ir  = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir

def convolve_wetdry(dry, ir, wet=0.7):
    wet_sig = signal.fftconvolve(dry, ir, mode='full')[:len(dry)]
    mx = max(1e-12, np.max(np.abs(wet_sig)))
    wet_sig = wet_sig / mx * (np.max(np.abs(dry)) + 1e-12)
    return (1.0 - wet) * dry + wet * wet_sig

# ============== 粒化：更顺滑预设（减少杂质） ==============
def granular_blur(y, sr, grain_ms=220, overlap=0.90, density=1.0, jitter_semitones=1.0):
    L = int(sr*grain_ms/1000.0)
    hop = max(1, int(L*(1-overlap)))

    y_pad = np.concatenate([y, np.zeros(L, dtype=y.dtype)])
    out = np.zeros_like(y_pad, dtype=float)

    i = 0
    while i < len(y):
        g = y_pad[i:i+L]
        if len(g) < L:
            g = np.pad(g, (0, L-len(g)))
        steps = np.random.uniform(-jitter_semitones, jitter_semitones)
        if abs(steps) > 1e-3:
            try:
                g = librosa.effects.pitch_shift(y=g, sr=sr, n_steps=steps)
            except Exception:
                pass
        g *= np.hanning(len(g))
        if np.random.rand() < density:
            out[i:i+L] += g[:min(L, len(out)-i)]
        i += hop

    out = out[:len(y)]
    return normalize(out)

# ============== 可选后处理：降噪/噪声门 ==============
def try_denoise(y, sr, strength=0.7):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, prop_decrease=strength)
    except Exception:
        return y

def noise_gate(y, threshold=0.02):
    mask = np.abs(y) > threshold
    return y * mask

# ============== 小房间 IR & 击打铺放（修复 NameError） ==============
def small_room_ir(sr, seconds=0.16, decay=15.0, lp=4500):
    """非常短促的小房间脉冲响应：不拖尾，避免“机关枪”叠加形成长尾"""
    n = int(seconds*sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    env = np.exp(-decay*t)
    ir  = np.random.randn(n) * env
    ny = 0.5*sr
    b, a = signal.butter(4, lp/ny, btype='low')
    ir  = signal.lfilter(b, a, ir)
    return normalize(ir, 0.9)

def place_hits(total_len, sr, hit_times, hit_wave, gain=0.3):
    """把单个击打样本铺放到指定时间戳上（秒），自动防止越界"""
    out = np.zeros(total_len, dtype=float)
    w = hit_wave * gain
    L = len(hit_wave)
    for t in hit_times:
        idx = int(round(t * sr))
        if idx >= total_len:
            continue
        end = min(total_len, idx + L)
        seg = end - idx
        out[idx:end] += w[:seg]
    return out

# ============== DRUM MORPH：把整轨“鼓腔化”，而不是逐点触发 ==============

def envelope_follow(y, sr, win_ms=20):
    n = max(1, int(sr*win_ms/1000))
    env = np.abs(y)
    kernel = np.ones(n)/n
    env = np.convolve(env, kernel, mode="same")
    m = np.max(env) + 1e-12
    return env / m

def resonant_bank(y, sr, centers=(70, 120, 200), Q=10):
    out = np.zeros_like(y, dtype=float)
    ny = 0.5*sr
    for f0 in centers:
        w0 = f0/ny
        b, a = iirpeak(w0, Q=Q)
        out += lfilter(b, a, y)
    mx = np.max(np.abs(out)) + 1e-12
    return out / mx

def synth_kick(sr, duration=0.25, f_start=160.0, f_end=50.0, amp=0.9, click=0.25):
    n = int(sr * duration)
    t = np.arange(n) / sr
    k = math.log(f_end / f_start) / duration
    inst_freq = f_start * np.exp(k * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    body = np.sin(phase)
    env = np.exp(-6.0 * t)
    kick = amp * body * env
    nc = max(1, int(0.008 * sr))
    click_noise = np.random.randn(nc) * click
    click_noise *= np.hanning(nc)
    out = kick.copy()
    out[:nc] += click_noise
    return normalize(out, 0.95)

def drum_morph(y, sr,
               centers=(70, 120, 200),
               Q=12,
               env_amt=0.6,
               mix=0.25,
               post_lowpass_hz=3500,
               drive=0.0):
    env = envelope_follow(y, sr, win_ms=20)
    res = resonant_bank(y, sr, centers=centers, Q=Q)
    res_mod = res * (1.0*(1-env_amt) + env_amt*env)
    ny = 0.5*sr
    b, a = signal.butter(2, post_lowpass_hz/ny, btype='low')
    res_mod = signal.lfilter(b, a, res_mod)
    if drive > 0:
        res_mod = np.tanh((1.0+drive)*res_mod)
    out = (1.0 - mix)*y + mix*normalize(res_mod, 0.95)
    return normalize(out, 0.95)

# ============== 节拍/稀疏击打层 ==============

def beat_grid_times(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return tempo, beats

def sparse_downbeats(beat_times, every=2, offset=0):
    picks = []
    for i, t in enumerate(beat_times):
        if (i - offset) % every == 0:
            picks.append(t)
    return np.array(picks)

def eq_lowshelf(y, sr, f0=90, gain_db=3.0, order=2):
    ny = 0.5*sr
    w = f0/ny
    b, a = signal.butter(order, w, btype='low')
    low = signal.lfilter(b, a, y)
    g = 10**(gain_db/20.0)
    return normalize((g*low + y) / (1+g), 0.99)

def one_knob_comp(y, sr, thresh_db=-14.0, ratio=3.0, makeup_db=3.0, win_ms=12):
    n = max(1, int(sr*win_ms/1000))
    rms = np.sqrt(np.convolve(y*y, np.ones(n)/n, mode='same') + 1e-12)
    rms_db = 20*np.log10(rms + 1e-12)
    over = np.maximum(0.0, rms_db - thresh_db)
    gain_db = - over * (1 - 1/ratio)
    lin = 10**((gain_db + makeup_db)/20.0)
    return normalize(y * lin, 0.99)

# （可选）很轻的 sidechain duck，默认不用，避免“抽水感”
def sidechain_duck(y, sr, hit_times, depth_db=3.0, pre_ms=5, atk_ms=15, hold_ms=80, rel_ms=120):
    depth = 10**(-depth_db/20.0)
    env = np.ones(len(y), dtype=float)
    def ms2samp(ms):
        return max(1, int(sr*ms/1000.0))
    pre = ms2samp(pre_ms)
    atk = ms2samp(atk_ms)
    hold = ms2samp(hold_ms)
    rel = ms2samp(rel_ms)
    for t in hit_times:
        idx = int(round(t*sr))
        a0 = max(0, idx - pre)
        a1 = min(len(y), a0 + atk)
        h1 = min(len(y), a1 + hold)
        r1 = min(len(y), h1 + rel)
        if a0 >= len(y):
            continue
        # 线性包络
        if a1 > a0: env[a0:a1] *= np.linspace(1.0, depth, a1-a0)
        if h1 > a1: env[a1:h1] *= depth
        if r1 > h1: env[h1:r1] *= np.linspace(depth, 1.0, r1-h1)
    return normalize(y * env, 0.99)

# ============== 主流程 ==============

def process(infile, outfile,
            pitch_steps=-8,
            rate=0.97,
            grain_ms=220, overlap=0.90, density=1.0, jitter=1.0,
            ir_sec=4.0, ir_decay=2.3, ir_lp=4000, wet=0.7,
            final_lp=6500,
            use_gate=False, gate_th=0.02,
            use_denoise=True, denoise_strength=0.7,
            drumify=True,
            drum_mix=0.25,
            drum_centers=(60, 90, 150),
            drum_Q=14,
            drum_env_amt=0.6,
            drum_drive=0.15,
            sparse_every=4, sparse_offset=0,   # 稀疏击打：每几拍一次
            sparse_gain=0.30                   # 击打层电平（低！）
            ):
    # 读入
    y, sr = librosa.load(infile, sr=None, mono=True)

    # 1) 降调
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_steps)

    # 2) 轻微变速
    try:
        y = librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        pass

    # 3) 粒化模糊（顺滑）
    y = granular_blur(y, sr, grain_ms=grain_ms, overlap=overlap, density=density, jitter_semitones=jitter)

    # 3.5) 预低通
    y = lowpass(y, sr, cutoff=7000, order=2)

    # 4) 卷积混响
    ir = make_ir(sr, seconds=ir_sec, decay=ir_decay, lp=ir_lp)
    y  = convolve_wetdry(y, ir, wet=wet)

    # 5) 末端 EQ
    y = lowpass(y, sr, cutoff=final_lp, order=4)

    # 6) 噪声门（可选）
    if use_gate:
        y = noise_gate(y, threshold=gate_th)

    # 7) 谱减降噪（可选）
    if use_denoise:
        y = try_denoise(y, sr, strength=denoise_strength)

    # 8) 鼓声化主层（连续“鼓腔化”）
    if drumify:
        y = drum_morph(
            y, sr,
            centers=drum_centers,
            Q=drum_Q,
            env_amt=drum_env_amt,
            mix=drum_mix,
            post_lowpass_hz=3500,
            drive=drum_drive
        )

        # 8.5) 稀疏击打层（非机关枪关键）
        try:
            tempo, beats = beat_grid_times(y, sr)
            if len(beats) > 0 and sparse_every is not None and sparse_every >= 1:
                hits = sparse_downbeats(beats, every=sparse_every, offset=sparse_offset)
                kick = synth_kick(sr, duration=0.22, f_start=150.0, f_end=48.0, amp=0.8, click=0.18)
                rir  = small_room_ir(sr, seconds=0.16, decay=15.0)
                kick = signal.fftconvolve(kick, rir, mode="full")[:len(kick)]
                kick = normalize(kick, 0.95)

                kick_track = place_hits(len(y), sr, hits, kick, gain=sparse_gain)
                kick_track = eq_lowshelf(kick_track, sr, f0=90, gain_db=3.0)
                kick_track = one_knob_comp(kick_track, sr, thresh_db=-18, ratio=2.5, makeup_db=2.0, win_ms=10)

                # 默认不 duck，避免抽水感；如需更干净可打开
                # y = sidechain_duck(y, sr, hits, depth_db=3.5, pre_ms=5, atk_ms=12, hold_ms=80, rel_ms=120)
                y = normalize(y + kick_track, 0.98)
        except Exception:
            # 如果节拍检测失败，就跳过稀疏层
            pass

    # 9) 归一 & 写文件
    y = normalize(y)
    sf.write(outfile, y, sr, subtype='FLOAT')
    print("Wrote", outfile)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        infile, outfile = sys.argv[1], sys.argv[2]
    else:
        infile  = r"TestAudioInput/1.m4a"
        outfile = r"TestAudioOutput/4_abstract.wav"
    process(infile, outfile)
