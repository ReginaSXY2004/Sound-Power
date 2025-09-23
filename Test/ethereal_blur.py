# ethereal_blur.py
# Minimal: make speech abstract/ethereal (pitch down + granular blur + long reverb)
# Usage:
#   python ethereal_blur.py "C:\Users\Regina Sun\文档\声音作品集\Test音频\1.m4a" "C:\Users\Regina Sun\文档\声音作品集\Test音频\1_abstract.wav"

import sys, numpy as np
import librosa, soundfile as sf
from scipy import signal

def normalize(x, peak=0.95):
    x = x / (np.max(np.abs(x)) + 1e-12)
    return x * peak

def lowpass(x, sr, cutoff=8000, order=4):
    ny = 0.5 * sr
    b, a = signal.butter(order, cutoff/ny, btype='low')
    return signal.lfilter(b, a, x)

def make_ir(sr, seconds=4.5, decay=2.6, lp=6000):
    n = int(seconds*sr)
    t = np.linspace(0, seconds, n)
    env = np.exp(-decay*t)
    ir  = np.random.randn(n) * env
    # 温暖一些：低通一点
    ny = 0.5*sr
    b,a = signal.butter(6, lp/ny, btype='low')
    ir  = signal.lfilter(b,a,ir)
    ir  = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir

def convolve(x, ir):
    y = signal.fftconvolve(x, ir, mode='full')[:len(x)]
    return y

def granular_blur(y, sr, grain_ms=140, overlap=0.7, density=0.95, jitter_semitones=3.0):
    L = int(sr*grain_ms/1000.0)
    hop = max(1, int(L*(1-overlap)))
    out = np.zeros_like(y, dtype=float)
    i = 0
    while i < len(y):
        g = y[i:i+L]
        if len(g) < L:
            g = np.pad(g, (0, L-len(g)))
        # 细微随机变调，破坏语素清晰度
        steps = np.random.uniform(-jitter_semitones, jitter_semitones)
        try:
            g = librosa.effects.pitch_shift(g, sr, steps)
        except Exception:
            pass
        g *= np.hanning(len(g))
        # 密度控制：偶尔跳过一些粒子
        if np.random.rand() < density:
            out[i:i+L] += g[:min(L, len(out)-i)]
        i += hop
    return normalize(out)

def process(infile, outfile):
    # 兼容 m4a/mp3/wav：优先 librosa（带 ffmpeg）
    y, sr = librosa.load(infile, sr=None, mono=True)
    # 1) 降调（权威/空灵的底色）——一整八度；想更空灵可到 -14
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-12)

    # 2) 轻微拉长（更庄严，避免过度）
    try:
        y = librosa.effects.time_stretch(y, rate=0.95)
    except Exception:
        pass
    # 3) 粒化模糊（核心的“听不清”）
    y = granular_blur(y, sr, grain_ms=160, overlap=0.7, density=0.95, jitter_semitones=3.0)
    # 4) 长混响（教堂感/空间感）
    ir = make_ir(sr, seconds=5.0, decay=2.8, lp=5500)
    y = convolve(y, ir)
    # 5) 轻低通（削辅音高频，进一步抽象）
    y = lowpass(y, sr, cutoff=7000, order=4)
    y = normalize(y)
    sf.write(outfile, y, sr, subtype='FLOAT')
    print("Wrote", outfile)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        infile, outfile = sys.argv[1], sys.argv[2]
    else:
        # 默认路径（方便直接 Run）
        infile  = r"TestAudioInput\1.m4a"
        outfile = r"TestAudioOutput\1_abstract.wav"
    process(infile, outfile)
