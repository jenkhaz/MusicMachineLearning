from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import normalize
import os

def process_audio(input_file, output_folder):
    # Load the MP3 file
    audio = AudioSegment.from_file(input_file, format="mp3")

    # Normalize the audio
    normalized_audio = normalize(audio)

    # Trim or extend to 30 seconds
    target_duration = 30 * 1000  # 30 seconds in milliseconds
    if len(normalized_audio) > target_duration:
        trimmed_audio = normalized_audio[:target_duration]
    else:
        silence = AudioSegment.silent(duration=target_duration - len(normalized_audio))
        trimmed_audio = normalized_audio + silence

    # Convert to WAV
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".wav")
    trimmed_audio.export(output_file, format="wav")
    print(f"Processed and saved: {output_file}")

def process_all_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            process_audio(os.path.join(input_folder, file), output_folder)

# Example usage
input_folder = "path_to_your_mp3_files"
output_folder = "path_to_save_wav_files"

process_all_files(input_folder, output_folder)
