from moviepy.editor import AudioFileClip

# MP4에서 WAV로 변환
def convert_mp4_to_wav(mp4_file, wav_file):
    audio = AudioFileClip(mp4_file)
    audio.write_audiofile(wav_file)

convert_mp4_to_wav("C:/Users/user1/Downloads/A.mp4", "C:/Users/user1/Downloads/A.wav")
convert_mp4_to_wav("C:/Users/user1/Downloads/D.mp4", "C:/Users/user1/Downloads/D.wav")
