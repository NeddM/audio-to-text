# audio-to-text
Action to parse audio to text

## Usage
```
- name: Audio to text
  uses: NeddM/audio-to-text@v1
  with:
    model: google                 # google or openai
    audio_file: path/to/file.mp3  # mp3 or wav files
    output_file: path/to/example.txt
    delete_audio_file: true       # optional, default false
```

Remember, you have to push changes later if you want it.