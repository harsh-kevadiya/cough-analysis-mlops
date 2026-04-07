import os
import subprocess

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed/audio_wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for file in os.listdir(INPUT_DIR):
    if file.endswith(".webm"):
        input_path = os.path.join(INPUT_DIR, file)
        output_file = file.replace(".webm", ".wav")
        output_path = os.path.join(OUTPUT_DIR, output_file)

        command = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",   # resample to 16kHz
            "-ac", "1",       # mono audio
            output_path
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            count += 1
        except:
            print(f"Error converting {file}")

print(f"✅ Converted {count} files to WAV format!")