# ethereal_blur.py  — clean ethereal + optional denoise + wet/dry reverb
# 用法：
#   python ethereal_blur.py "TestAudioInput\1.m4a" "TestAudioOutput\1_abstract.wav"

import sys, numpy as np
import librosa, soundfile as sf
from scipy import signal

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
    # 暖一点：低通降噪
    ny = 0.5*sr
    b,a = signal.butter(6, lp/ny, btype='low')
    ir  = signal.lfilter(b,a,ir)
    ir  = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir

def convolve_wetdry(dry, ir, wet=0.7):
    """卷积混响 + 湿干比；不要全湿以免放大杂质"""
    wet_sig = signal.fftconvolve(dry, ir, mode='full')[:len(dry)]
    mx = max(1e-12, np.max(np.abs(wet_sig)))
    # 匹配干声能量，防止一声变巨响
    wet_sig = wet_sig / mx * (np.max(np.abs(dry)) + 1e-12)
    return (1.0 - wet) * dry + wet * wet_sig

# ============== 粒化：更顺滑预设（减少杂质） ==============
def granular_blur(y, sr, grain_ms=220, overlap=0.90, density=1.0, jitter_semitones=1.0):
    """
    更平滑的粒化：
      - 更长的粒子 + 更高重叠 => 更少接缝/毛边
      - 更小的随机变调 => 避免金属颤
      - 不丢粒（density=1.0） => 避免咔哒
    """
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
                g = librosa.effects.pitch_shift(y=g, sr=sr, n_steps=steps)  # 兼容 librosa>=0.10
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
    """
    频谱减法降噪（需要 noisereduce；没安装会自动跳过）
    pip install noisereduce
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, prop_decrease=strength)
    except Exception:
        return y

def noise_gate(y, threshold=0.02):
    """简单噪声门：能量低于阈值直接压下去"""
    mask = np.abs(y) > threshold
    return y * mask

# ============== 主流程 ==============
def process(infile, outfile,
            pitch_steps=-8,            # 比 -12 更自然，含蓄
            rate=0.97,                 # 微慢
            grain_ms=220, overlap=0.90, density=1.0, jitter=1.0,
            ir_sec=4.0, ir_decay=2.3, ir_lp=4000, wet=0.7,
            final_lp=6500,
            use_gate=False, gate_th=0.02,
            use_denoise=True, denoise_strength=0.7):
    # 读 m4a/mp3/wav：librosa + ffmpeg 兜底
    y, sr = librosa.load(infile, sr=None, mono=True)

    # 1) 降调（幽深/权威底色）
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_steps)

    # 2) 轻微变速
    try:
        y = librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        pass

    # 3) 粒化模糊（更顺滑预设）
    y = granular_blur(y, sr, grain_ms=grain_ms, overlap=overlap, density=density, jitter_semitones=jitter)

    # 3.5) 预清理：轻低通去齿音毛边（放在卷积前）
    y = lowpass(y, sr, cutoff=7000, order=2)

    # 4) 合成 IR + 湿干比卷积（空间感但不过分放大噪声）
    ir = make_ir(sr, seconds=ir_sec, decay=ir_decay, lp=ir_lp)
    y  = convolve_wetdry(y, ir, wet=wet)

    # 5) 末端 EQ
    y = lowpass(y, sr, cutoff=final_lp, order=4)

    # 6) 可选：噪声门（切掉拖太久的细尾巴）
    if use_gate:
        y = noise_gate(y, threshold=gate_th)

    # 7) 可选：谱减法降噪（保留混响主体，刮掉嘶声底噪）
    if use_denoise:
        y = try_denoise(y, sr, strength=denoise_strength)

    # 8) 归一
    y = normalize(y)
    sf.write(outfile, y, sr, subtype='FLOAT')
    print("Wrote", outfile)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        infile, outfile = sys.argv[1], sys.argv[2]
    else:
        # 默认：方便直接点 ▶ 运行
        infile  = r"TestAudioInput\1.m4a"
        outfile = r"TestAudioOutput\3_abstract.wav"
    process(infile, outfile)
