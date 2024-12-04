import os
import numpy as np
import soundfile as sf

def process_audio(input_file, output_folder):
    # Load the audio file
    audio, samplerate = sf.read(input_file)

    # Normalize the audio
    normalized_audio = audio / np.max(np.abs(audio))

    # Trim or extend to 30 seconds
    target_duration = 30  # 30 seconds
    current_duration = len(normalized_audio) / samplerate
    if current_duration > target_duration:
        trimmed_audio = normalized_audio[:int(target_duration * samplerate)]
    else:
        padding = np.zeros(int((target_duration - current_duration) * samplerate))
        trimmed_audio = np.concatenate((normalized_audio, padding))

    # Convert to WAV
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".wav")
    sf.write(output_file, trimmed_audio, samplerate)
    print(f"Processed and saved: {output_file}")

def process_all_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".mp3") or file.endswith(".wav"):
            process_audio(os.path.join(input_folder, file), output_folder)

# Example usage
input_folder = r"C:\Users\User\OneDrive - American University of Beirut\Desktop\E3\EECE 490\MLproj\Data\Techno"
output_folder = r"C:\Users\User\OneDrive - American University of Beirut\Desktop\E3\EECE 490\MLproj\Data\Tech"

process_all_files(input_folder, output_folder)
