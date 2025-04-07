import requests
import json

# Define the URL for your FastAPI server
url = "http://localhost:8000/transcribe/"  # Adjust if your server uses a different port

# Open the files from the test_video folder
try:
    audio_file = open("test_video/audio.wav", "rb")
    video_file = open("test_video/test.mp4", "rb")
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit(1)

# Define the parameters as a dictionary
params_dict = {
    "language": "en",
    "noise_snr": 0,  # Use integer if the server expects it; otherwise, keep as "0"
    "task": "transcribe",
    "modalities": "avsr",
    "beam_size": 1,  # Use integer if the server expects it; otherwise, keep as "1"
    "fp16": 0,  # Use integer if the server expects it; otherwise, keep as "0"
    "checkpoint_path": "models/whisper-flamingo_en-x_small.pt",
    "noise_fn": "noise/babble/muavic/test.tsv"
}

# Convert the parameters dictionary to a JSON string
params_json = json.dumps(params_dict)

# Prepare the files and form data
files = {
    "audio_file": audio_file,
    "video_file": video_file
}
data = {
    "params": params_json  # Send all parameters as a single JSON string
}

# Send the POST request with files and data
response = requests.post(url, files=files, data=data)

# Print response details
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

# Safely handle the response
if response.status_code == 200:
    try:
        transcription = response.json().get("text", "No transcription found")
        print(f"Transcription: {transcription}")
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON")
else:
    print(f"Server error {response.status_code}: {response.text}")

# Close the files to free resources
audio_file.close()
video_file.close()