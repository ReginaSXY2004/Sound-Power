# -*- coding: utf-8 -*-
"""
oneclick_fix.py  (放在 Test 目录下)

流程：
  1) 自动找到 record_input 中最新录音
  2) 用 Test/ethereal_blur.py 处理 → TestAudioOutput/2_abstract.wav
  3) 用 Test/Random3-6Par.py 随机修改 3–6 个参数 → 修改完参数Output/*.wav
"""

import os
import sys
import glob
import subprocess

# ============= 1. 项目根目录 =============
# oneclick_fix.py 位于:  .../Sound-Power/Test/oneclick_fix.py
# 所以 Test 文件夹的上一级就是项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ============= 2. 各类目录 =============
RECORD_DIR   = os.path.join(BASE_DIR, "record_input")

TEST_DIR     = os.path.join(BASE_DIR, "Test")
TEST_OUTPUT  = os.path.join(BASE_DIR, "TestAudioOutput")
MODIFIED_DIR = os.path.join(BASE_DIR, "修改完参数Output")

# ============= 3. 脚本路径 =============
ETHEREAL_PY   = os.path.join(TEST_DIR, "ethereal_blur.py")
RANDOM_FIX_PY = os.path.join(TEST_DIR, "Random3-6Par.py")

# ============= 函数 =============
def get_latest_recording():
    """找到 record_input 中最新的 wav 文件"""
    pattern = os.path.join(RECORD_DIR, "*.wav")
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"record_input 里没有发现录音，请先运行 record_response.py 录一条。")
    latest = max(files, key=os.path.getmtime)
    return latest


def run_ethereal_blur(infile, outfile):
    """调用 ethereal_blur.py"""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    print(f"[1/2] 运行 ethereal_blur.py:")
    print(f"     输入: {infile}")
    print(f"     输出: {outfile}")

    cmd = [sys.executable, ETHEREAL_PY, infile, outfile]
    subprocess.check_call(cmd)

    print("[1/2] ethereal_blur 完成。")


def run_random_3_6():
    """调用 Random3-6Par.py（其内部已固定读取 TestAudioOutput/2_abstract.wav）"""
    print("[2/2] Random3-6Par：随机修改 3–6 个参数...")

    cmd = [sys.executable, RANDOM_FIX_PY]
    subprocess.check_call(cmd)

    print("[2/2] Random3-6Par 完成。")


# ============= 主函数 =============
def main():
    # ---- (A) 找录音 ----
    if len(sys.argv) >= 2:
        src_audio = sys.argv[1]
        if not os.path.isfile(src_audio):
            raise SystemExit(f"[Error] 指定音频不存在：{src_audio}")
        print(f"[*] 使用指定录音: {src_audio}")
    else:
        src_audio = get_latest_recording()
        print(f"[*] 自动使用最新录音: {src_audio}")

    # ---- (B) 空灵处理 ----
    abstract_path = os.path.join(TEST_OUTPUT, "2_abstract.wav")
    run_ethereal_blur(src_audio, abstract_path)

    # ---- (C) 修改 3–6 参数 ----
    run_random_3_6()

    print("\n=====================================================")
    print("  🎉 OneClickFix 完成！")
    print("  输入录音:", src_audio)
    print("  中间文件:", abstract_path)
    print("  最终输出目录:", MODIFIED_DIR)
    print("  请在 TouchDesigner 的第二个 visual 中播放最终输出文件")
    print("=====================================================")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("[Subprocess Error]", e)
        sys.exit(1)
    except Exception as e:
        print("[Error]", e)
        sys.exit(1)