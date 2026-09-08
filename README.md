# AI Voice Studio

A Streamlit-based voice generation application powered by **OmniVoice**. The app allows you to upload a reference voice, provide its transcript, select a language and emotion, adjust generation parameters, and generate speech in the reference voice.

## Features

- 🎙️ **Voice cloning** using an uploaded reference recording
- 🌍 Multiple language options:
  - English
  - Hindi
  - Bengali
  - Spanish
  - French
  - German
  - Japanese
  - Chinese
  - Arabic
- 🎭 Emotion/style presets
- ⚡ Adjustable speaking speed
- 🧠 Adjustable generation steps
- 🎯 Adjustable guidance scale
- 🔊 In-browser audio playback
- ⬇️ Download generated audio as WAV
- 🌙 Custom dark-themed Streamlit interface
- 🩵 Custom `#0bd9e0` slider styling
- 🖥️ Automatic CUDA/CPU selection

## How It Works

The application follows this workflow:

1. Load the `k2-fsa/OmniVoice` model.
2. Detect whether CUDA is available.
3. Upload a reference voice recording.
4. Enter the exact transcript corresponding to the reference recording.
5. Select the desired language and emotion.
6. Adjust speaking speed and generation parameters.
7. Generate the speech using OmniVoice.
8. Play the generated audio in the browser.
9. Download the generated WAV file.

## Requirements

The main Python libraries used by the application are:

- `streamlit`
- `torch`
- `numpy`
- `soundfile`
- `omnivoice`

Python 3.10+ is recommended.

## Installation

### 1. Clone or download the project

Place the Python application file in your project directory.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main packages:

```bash
pip install streamlit numpy soundfile torch omnivoice
```

For NVIDIA GPU users, install the appropriate CUDA-enabled PyTorch build for your installed CUDA/driver environment.

## Running the Application

Start Streamlit with:

```bash
streamlit run app.py
```

Replace `app.py` with the actual name of your Python file if it is different.

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser.

## GPU Support

The application automatically checks for CUDA:

```python
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
```

When a compatible NVIDIA GPU is available, OmniVoice is loaded on CUDA using `float16`.

To verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

To see the detected GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Reference Voice

For best results:

- Use a clean speech recording.
- A reference recording of approximately **3–10 seconds** is recommended by the application.
- Avoid loud background noise.
- Avoid music or multiple speakers.
- Provide a transcript that closely matches what is actually spoken in the recording.

The reference transcript is important because it is passed to OmniVoice together with the reference audio.

## Emotion Controls

The application provides the following emotion presets:

| Emotion | OmniVoice Instruction |
|---|---|
| Neutral | moderate pitch |
| Happy | high pitch |
| Sad | low pitch |
| Angry | high pitch |
| Excited | very high pitch |
| Calm | low pitch |
| Serious | moderate pitch |
| Warm | low pitch |
| Friendly | moderate pitch |
| Confident | moderate pitch |
| Dramatic | high pitch |
| Fearful | very high pitch |
| Whispering | whisper |

These presets are implemented through the `EMOTION_INSTRUCTIONS` dictionary in the application.

Note that these are **style instructions**, not separately trained emotion models. The actual expressive result depends on OmniVoice and the reference voice.

## Generation Settings

### Speaking Speed

Range:

```text
0.6x – 1.5x
```

- Below `1.0` → slower speech
- `1.0` → normal speed
- Above `1.0` → faster speech

### Generation Steps

Range:

```text
16 – 64
```

Higher values may improve generation quality but increase generation time.

### Guidance Scale

Range:

```text
0.5 – 5.0
```

This controls how strongly generation follows the conditioning information.

## UI Customization

The application uses custom CSS for its dark interface.

The slider accent color is customized to:

```text
#0bd9e0
```

The CSS can be modified in the `CUSTOM CSS` section of the Python file if you want to change the application's appearance.

## Output

Generated speech is saved temporarily as a WAV file with:

- Format: WAV
- Sample rate: 24,000 Hz

The application provides both browser playback and a download button.

The downloaded file is named:

```text
omnivoice_output.wav
```

## Project Structure

A simple project structure can be:

```text
ai-voice-studio/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── config.toml
```

The `.streamlit/config.toml` file is optional.

## Voice Usage Consent

The application includes a consent checkbox:

> I have permission to clone/use the uploaded voice.

Generation is blocked unless the user confirms that they have permission to use the reference voice.

Only use voice recordings that you have the legal right or permission to clone.

## Troubleshooting

### OmniVoice fails to load

Check that the required packages are installed:

```bash
pip install -r requirements.txt
```

Also make sure your Python, PyTorch, CUDA, and GPU environment are compatible.

### CUDA is not detected

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns:

```text
False
```

PyTorch is currently running without CUDA support or cannot access your NVIDIA GPU.

### Generation is very slow

Possible causes include:

- Running on CPU
- Limited GPU VRAM
- Large generation step count
- Model loading overhead

Try lowering **Generation steps** if quality remains acceptable.

### Poor voice similarity

Try:

- Using a cleaner reference recording
- Using a 3–10 second reference
- Providing an accurate transcript
- Removing background noise
- Using a reference voice with clear speech

### Emotion does not sound strong enough

The emotion selector maps to OmniVoice's supported style instructions. The selected emotion does not guarantee a specific emotional performance. Experiment with the reference recording, text, and generation settings.

## Security and Privacy

The application processes uploaded reference audio locally when running the OmniVoice model locally. Temporary audio files are created during processing.

Do not upload voice recordings unless you have permission to use them.

## Credits

Powered by:

**OmniVoice — `k2-fsa/OmniVoice`**

Application UI:

**Streamlit**

## License

This README does not define a license for OmniVoice or any third-party dependency. Check the respective project licenses before distributing or commercializing the application.
