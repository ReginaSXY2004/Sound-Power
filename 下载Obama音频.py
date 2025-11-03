# 下载后会生成 “Barack Obama...” 一类文件名
# 进入Python交互环境或写个脚本截取片段：
from pydub import AudioSegment

audio = AudioSegment.from_file("President Obama's best speeches.wav")
clip = audio[20_000:40_000]  # 毫秒
clip.export(r"C:\Users\Regina Sun\Documents\GitHub\Sound-Power\TestAudioInput\Obama.wav", format="wav")
