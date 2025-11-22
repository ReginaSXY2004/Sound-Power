# -*- coding: utf-8 -*-
"""
record_response.py
按 R 开始录音，按 Q 停止。保存到 RECORD_DIR，然后调用 karen_score_tiered_v2.py 分析，
最后根据结果更新 ui_state.json。要不要 OneClickFix 由 TouchDesigner 里的按钮决定：
TD 修改 ui_state.json 里的 fix_choice（'fix' / 'skip'），本脚本轮询等待。
"""

import os
import sys
import time
import threading
import datetime
import subprocess
import numpy as np
import json

# ====== 路径配置（根据你的目录结构固定写死）======
REPO_ROOT   = r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power"
ANALYZER_PY = os.path.join(REPO_ROOT, "Test", "karen_score_tiered_v2.py")

RECORD_DIR  = r"C:\Users\Regina Sun\Documents\声音作品集\Record_Voice"   # ✅ 录音文件放这里

# TD 读取 / 修改的 UI 状态 JSON（注意这里改成 record_input 下面）
STATE_PATH  = os.path.join(REPO_ROOT, "record_input", "ui_state.json")

# oneclick_fix.py 的路径
ONECLICK_PY = os.path.join(REPO_ROOT, "Test", "oneclick_fix.py")

# ====== 录音配置（与分析脚本保持一致）======
SR       = 16000        # 采样率
CHANNELS = 1
DTYPE    = "float32"

# ====== 依赖检查 ======
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[Error] 缺少依赖：sounddevice / soundfile")
    print("请先执行：pip install sounddevice soundfile")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("[Error] 缺少依赖：pynput")
    print("请先执行：pip install pynput")
    sys.exit(1)


# ====== UI 状态读写 ======
def read_state():
    """读取 ui_state.json，没有就返回默认。"""
    state = {
        "scene": "live",
        "latest_record": "",
        "latest_random": "",
        "fix_choice": ""
    }
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                state.update(old)
        except Exception as e:
            print("[UI] 读 ui_state.json 失败：", e)
    return state


def write_state(scene=None, latest_record=None, latest_random=None, fix_choice=None):
    """
    更新 ui_state.json，告诉 TouchDesigner 目前处于哪个场景。
    - scene: 'live' / 'score' / 'fixed'，如果是 None 就不改
    - latest_record: 最近一次原始录音路径（可选）
    - latest_random: 最近一次随机化音频路径（目前 TD 自己找最新文件，这里可以先留空或以后再填）
    - fix_choice: '', 'fix', 'skip'
    """
    state = read_state()

    if scene is not None:
        state["scene"] = scene
    if latest_record is not None:
        state["latest_record"] = latest_record
    if latest_random is not None:
        state["latest_random"] = latest_random
    if fix_choice is not None:
        state["fix_choice"] = fix_choice

    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[UI] state -> {state}")
    except Exception as e:
        print("[UI] 写 ui_state.json 失败：", e)


def wait_for_fix_choice(poll_interval=0.2):
    """
    分析结束后，处于 score 场景，等待 TouchDesigner 那边的按钮选择：
    - fix_choice == 'fix'  → 返回 'fix'
    - fix_choice == 'skip' → 返回 'skip'
    其它值或空字符串则继续等待。
    """
    print("\n[Wait] 等待 TouchDesigner 里的 OneClickFix 选择（fix / skip）...")
    print("      （观众在屏幕上点按钮，你这里不用操作。）")
    try:
        while True:
            state = read_state()
            choice = state.get("fix_choice", "")
            if choice in ("fix", "skip"):
                print(f"[Wait] 检测到选择: {choice}")
                return choice
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[Wait] 检测中断，返回 skip。")
        return "skip"


# ====== 录音部分 ======
def record_until_q(sr=SR, channels=CHANNELS, out_dir=RECORD_DIR):
    """
    按 R 开始录音，按 Q 停止录音。返回保存的 wav 路径。
    """
    os.makedirs(out_dir, exist_ok=True)

    print("\n[Record Mode]")
    print("提示：按 R 开始录音，按 Q 停止录音。\n")

    start_ev = threading.Event()
    stop_ev  = threading.Event()
    frames   = []

    def on_press(key):
        try:
            k = key.char.lower()  # 字母键
        except AttributeError:
            k = str(key).lower()
        if not start_ev.is_set() and k in ("r",):
            print(">> 开始录音… (按 Q 停止)")
            start_ev.set()
        elif start_ev.is_set() and k in ("q",):
            print(">> 停止录音")
            stop_ev.set()
            return False  # 结束监听

    def callback(indata, frames_count, time_info, status):
        if status:
            print("[SD status]", status)
        frames.append(indata.copy())

    # 键盘监听（异步）
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 等待 R
    start_ev.wait()

    # 采集音频直到 Q
    with sd.InputStream(samplerate=sr, channels=channels, dtype=DTYPE, callback=callback):
        while not stop_ev.is_set():
            time.sleep(0.05)

    # 拼接并保存
    if not frames:
        raise RuntimeError("没有采集到音频数据")
    wav = np.concatenate(frames, axis=0)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = os.path.join(out_dir, f"Recorded_{ts}.wav")
    sf.write(out_wav, wav, sr)
    print(f">> 已保存：{out_wav}")
    return out_wav


# ====== 调用分析脚本 ======
def run_analyzer(audio_path):
    """
    调用 karen_score_tiered_v2.py 对指定音频做分析。
    使用当前 Python 解释器（保持环境一致）。
    """
    if not os.path.isfile(ANALYZER_PY):
        print(f"[Error] 分析脚本未找到：{ANALYZER_PY}")
        return 1

    cmd = [sys.executable, ANALYZER_PY, audio_path]
    print("\n[Run Analyzer]")
    print("CMD:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("-" * 72)

    # 直接输出到当前控制台
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            print(line, end="")
    finally:
        proc.stdout.close()
        ret = proc.wait()
    print("-" * 72)
    if ret == 0:
        print("[DONE] 分析完成。")
    else:
        print(f"[ERROR] 分析进程返回码：{ret}")
    return ret


# ====== 主流程 ======
def main():
    print("=== Karen Recorder ===")
    print("录音将保存到:", RECORD_DIR)
    print("准备就绪。")

    # 进程序时先告诉 TD：现在是 live 场景，清空 fix_choice
    write_state(scene="live", fix_choice="")

    try:
        wav_path = record_until_q(sr=SR, channels=CHANNELS, out_dir=RECORD_DIR)
    except Exception as e:
        print("[Record Error]", e)
        # 出错就维持在 live
        write_state(scene="live")
        sys.exit(2)

    # 录完，跑分析
    code = run_analyzer(wav_path)
    if code != 0:
        # 分析失败，也维持 live（或者以后你可以做一个 error 场景）
        write_state(scene="live", latest_record=wav_path)
        sys.exit(code)

    # 分析 OK：进入 “score” 状态（中间 UI：显示分数+文案）
    # fix_choice 先清空，让 TD 按钮去写
    write_state(scene="score", latest_record=wav_path, fix_choice="")

    # 等待 TouchDesigner 按钮选择 fix / skip
    choice = wait_for_fix_choice()

    if choice == "fix":
        # ===== 跑 OneClickFix =====
        if not os.path.isfile(ONECLICK_PY):
            print("[Error] 找不到 oneclick_fix.py：", ONECLICK_PY)
            # 失败就回到 live
            write_state(scene="live", latest_record=wav_path, fix_choice="")
            sys.exit(1)

        print("\n[OneClickFix] 正在运行 oneclick_fix.py ...\n")
        try:
            # 把这次录音的路径传给 oneclick_fix.py
            subprocess.check_call([sys.executable, ONECLICK_PY, wav_path])
        except Exception as e:
            print("[OneClickFix Error]", e)
            write_state(scene="live", latest_record=wav_path, fix_choice="")
            sys.exit(1)

        # oneclick_fix 完成，此时 Changed_Voice 里已经有最新 randomized 音频
        # TD 那边用 folder_latest/sort_timestamp 去找“最新文件”，这里只需要切场景
        write_state(scene="fixed", latest_record=wav_path, fix_choice="")
        print("\n[UI] 已切换为 fixed 场景，请看 TouchDesigner。")

    else:
        # 用户选择 skip：回到 live（或你以后做一个 “只看评分不修音” 的独立场景也行）
        write_state(scene="live", latest_record=wav_path, fix_choice="")
        print("\n[UI] 保持 live 场景。")


if __name__ == "__main__":
    main()
