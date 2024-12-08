import os
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf

def process_audio_files(classical_path, disco_path):
    if not os.path.exists(classical_path):
        print(f"File not found: {classical_path}")
        return
    if not os.path.exists(disco_path):
        print(f"File not found: {disco_path}")
        return

    # Load the tracks
    classical, sr1 = librosa.load(classical_path, sr=None)
    disco, sr2 = librosa.load(disco_path, sr=None)

    # Detect beats and tempos
    tempo1_array, beats1 = librosa.beat.beat_track(y=classical, sr=sr1)
    tempo2_array, beats2 = librosa.beat.beat_track(y=disco, sr=sr2)

    # Extract scalar tempo values
    tempo1 = tempo1_array if isinstance(tempo1_array, (int, float)) else tempo1_array[0]
    tempo2 = tempo2_array if isinstance(tempo2_array, (int, float)) else tempo2_array[0]

    print(f"Classical Tempo: {tempo1}, Disco Tempo: {tempo2}")

    # Adjust tempo of disco to match classical
    target_tempo = tempo1  # Classical tempo
    disco_matched = librosa.effects.time_stretch(disco, rate=tempo2 / target_tempo)


        # Align lengths: Match the length of disco to classical
    aligned_disco = librosa.util.fix_length(disco_matched, len(classical))

    # Save numpy arrays as temporary WAV files
    temp_classical = os.path.join(os.path.dirname(classical_path), "temp_classical.wav")
    temp_disco = os.path.join(os.path.dirname(disco_path), "temp_disco.wav")
    sf.write(temp_classical, classical, sr1)
    sf.write(temp_disco, aligned_disco, sr2)

    # Load these WAVs into pydub
    classical_segment = AudioSegment.from_file(temp_classical)
    disco_segment = AudioSegment.from_file(temp_disco)

    # Combine tracks
    combined = classical_segment.overlay(disco_segment)

    # Save fusion output in the same directory as disco
    fusion_output = os.path.join(os.path.dirname(disco_path), "fusion_output.wav")
    combined.export(fusion_output, format="wav")
    print(f"Fusion output saved as: {fusion_output}")

    # Optional: Pitch shift classical track
    classical_shifted = librosa.effects.pitch_shift(classical, sr=sr1, n_steps=-2)
    shifted_output = os.path.join(os.path.dirname(classical_path), "classical_shifted.wav")
    sf.write(shifted_output, classical_shifted, sr1)
    print(f"Pitch-shifted classical track saved as: {shifted_output}")

# Input file paths
classical_file = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\01.wav"
disco_file =r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\01 copy.wav"

# Process the audio files
process_audio_files(classical_file, disco_file)
