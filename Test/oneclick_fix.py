# -*- coding: utf-8 -*-
"""
oneclick_fix.py
保持你的原始逻辑：
  1) ethereal_blur 处理唯一输入
  2) Random3_6Par.py 做 3–6 声学修改
"""

import os
import sys
import glob
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RECORD_DIR   = os.path.join(BASE_DIR, "record_input")
TEST_DIR     = os.path.join(BASE_DIR, "Test")
TEST_OUTPUT  = os.path.join(BASE_DIR, "TestAudioOutput")
MODIFIED_DIR = os.path.join(BASE_DIR, "修改完参数Output")

ETHEREAL_PY   = os.path.join(TEST_DIR, "ethereal_blur.py")
RANDOM_FIX_PY = os.path.join(TEST_DIR, "Random3_6Par.py")

def get_latest_recording():
    pattern = os.path.join(RECORD_DIR, "*.wav")
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError("record_input 里没有录音")
    return max(files, key=os.path.getmtime)

def run_ethereal_blur(infile, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print("[1/2] ethereal_blur:", infile)
    cmd = [sys.executable, ETHEREAL_PY, infile, outfile]
    subprocess.check_call(cmd)

def run_random_3_6():
    print("[2/2] Random3-6Par...")
    cmd = [sys.executable, RANDOM_FIX_PY]
    subprocess.check_call(cmd)

def main():
    if len(sys.argv) >= 2:
        src_audio = sys.argv[1]
    else:
        src_audio = get_latest_recording()

    abstract_path = os.path.join(TEST_OUTPUT, "2_abstract.wav")
    run_ethereal_blur(src_audio, abstract_path)

    run_random_3_6()

    print("\n===== OneClickFix DONE =====")
    print("输入录音:", src_audio)
    print("中间文件:", abstract_path)
    print("修改后文件输出目录:", MODIFIED_DIR)
    print("TD 可读取的 six.json 也在这个目录里")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[Error]", e)
        sys.exit(1)
