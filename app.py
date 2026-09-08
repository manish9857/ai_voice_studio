import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from omnivoice import OmniVoice


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Voice Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 60% 50%,
                rgba(14, 115, 114, 0.20),
                transparent 50%
            ),
           #010d12;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 0.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;

        background:
            linear-gradient(
                145deg,
                rgba(1, 12, 15, 0.99),
                rgba(39, 245, 243, 0.40)
            );

        border: 1px solid rgba(14, 115, 114, 0.16);

        box-shadow:
            0 20px 60px rgba(8, 18, 25, 0.15);
    }

    .hero h1 {
        font-size: 2rem;
        margin: 0;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero p {
        color: #80989e;
        font-size: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin-bottom: 0.8rem;
        border-radius: 999px;

        background: rgba(1, 12, 15, 0.99);
        border: 1px solid rgba(39, 245, 243, 0.40);

        color: #c7d2fe;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* ---------- Generate button ---------- */

    div.stButton > button[kind="primary"] {
        width: 100%;
        height: 3.2rem;

        border-radius: 14px;
        border: none;

        font-size: 1rem;
        font-weight: 750;

        background: linear-gradient(
            0deg,
            rgba(1, 12, 15, 0.99),
            rgba(39, 245, 243, 0.2),
            rgba(1, 12, 15, 0.99)
        );

        color: white;

        box-shadow:
            0 10px 30px rgba(99, 102, 241, 0.30);

        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);

        box-shadow:
            0 14px 35px rgba(99, 102, 241, 0.30);
    }
    

    /* ---------- Text area ---------- */

    textarea {
        border-radius: 14px !important;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #080c14;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* ---------- Divider ---------- */

    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_id = "k2-fsa/OmniVoice"
    

    if torch.cuda.is_available():

        device = "cuda"

        # float16 is generally safer than bfloat16
        # for consumer NVIDIA GPUs.
        dtype = torch.float16

    else:

        device = "cpu"
        dtype = torch.float32

    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )

    return model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_audio(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    return temp_file.name

EMOTION_INSTRUCTIONS = {
    "Neutral": "moderate pitch",
    "Happy": "high pitch",
    "Sad": "low pitch",
    "Angry": "high pitch",
    "Excited": "very high pitch",
    "Calm": "low pitch",
    "Serious": "moderate pitch",
    "Warm": "low pitch",
    "Friendly": "moderate pitch",
    "Confident": "moderate pitch",
    "Dramatic": "high pitch",
    "Fearful": "very high pitch",
    "Whispering": "whisper",
    }


def generate_audio(
    model,
    text,
    ref_audio,
    ref_text,
    language,
    emotion,
    speed,
    guidance_scale,
    num_steps,
):

    # Build style/emotion instruction.
    #
    # OmniVoice supports "instruct" together with reference audio.
    #

    instruct = EMOTION_INSTRUCTIONS.get(
            emotion,
            "moderate pitch"
        )

    generation_kwargs = {
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "language": language,
        "speed": speed,
        "guidance_scale": guidance_scale,
        "num_step": num_steps,
    }

    if instruct:
        generation_kwargs["instruct"] = instruct

    audio = model.generate(**generation_kwargs)

    return audio[0]


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading OmniVoice model..."):

    try:
        model = load_model()

    except Exception as e:

        st.error("Could not load OmniVoice.")

        st.exception(e)

        st.stop()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="badge">
            AI VOICE STUDIO
        </div>

        <h1>🎙️ AI Voice Studio</h1>

        <p>
            Clone a voice, control its emotion, and generate
            natural-sounding speech from text.
        </p>

    </div>
    """
)

# st.markdown("""
# <style>
# /* Slider track */
# div[data-baseweb="slider"] > div > div {
#     background: red !important;
# }

# /* Slider thumb */
# div[data-baseweb="slider"] div[role="slider"] {
#     background-color: red !important;
#     border-color: red !important;
# }
# </style>
# """, unsafe_allow_html=True)


st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Generation Settings")

    st.divider()

    language = st.selectbox(
        "🌍 Language",
        [
            "English",
            "Hindi",
            "Bengali",
            "Spanish",
            "French",
            "German",
            "Japanese",
            "Chinese",
            "Arabic",
        ],
        index=0,
    )

    emotion = st.selectbox(
        "🎭 Emotion",

        [
            "Neutral",
            "Happy",
            "Sad",
            "Angry",
            "Excited",
            "Calm",
            "Serious",
            "Warm",
            "Friendly",
            "Confident",
            "Dramatic",
            "Fearful",
            "Whispering",
        ],

        index=0,

        help=(
            "Emotion is passed to OmniVoice as a style "
            "instruction together with the reference voice."
        ),
    )

    speed = st.slider(
        "⚡ Speaking speed",
        min_value=0.6,
        max_value=1.5,
        value=1.0,
        step=0.05,

        help=(
            "Values below 1.0 are slower. "
            "Values above 1.0 are faster."
        ),
    )

    num_steps = st.slider(
        "🧠 Generation steps",
        min_value=16,
        max_value=64,
        value=32,
        step=4,

        help=(
            "Higher values can improve generation quality "
            "but increase generation time."
        ),
    )

    guidance_scale = st.slider(
        "🎯 Guidance scale",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,

        help=(
            "Controls how strongly the generation follows "
            "the conditioning information."
        ),
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [1, 1.15],
    gap="large",
)


# ============================================================
# LEFT — VOICE
# ============================================================

with left:

    uploaded_audio = st.file_uploader(
        "Upload reference voice",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
        help=(
            "A clean 3–10 second speech recording "
            "usually works best."
        ),
    )

    reference_audio_path = None

    if uploaded_audio:

        reference_audio_path = save_uploaded_audio(
            uploaded_audio
        )

        st.audio(
            uploaded_audio,
            format=uploaded_audio.type,
        )

        st.success(
            f"Loaded: {uploaded_audio.name}"
        )

    st.write("")

    ref_text = st.text_area(
        "📝 Reference transcript",

        placeholder=(
            "Type exactly what is spoken in the reference audio..."
        ),

        height=120,

        help=(
            "The transcript should closely match the "
            "uploaded reference recording."
        ),
    )


# ============================================================
# RIGHT — TEXT + EMOTION
# ============================================================

with right:

    default_text = (
        "Once upon a time, there was a young man called Ma Liang. "
        "He was poor and kind and liked drawing so much that he "
        "drew pictures everywhere."
    )

    text_for_tts = st.text_area(
        "Text to generate",
        value=default_text,
        height=180,
    )

    st.write("")

# ============================================================
# CONSENT
# ============================================================

st.divider()

consent = st.checkbox(
    "I have permission to clone/use the uploaded voice.",
)


# ============================================================
# GENERATE
# ============================================================

generate = st.button(
    "✨ Generate Voice",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATION
# ============================================================

if generate:

    if not consent:

        st.warning(
            "Please confirm that you have permission to use "
            "the reference voice."
        )

        st.stop()

    if not uploaded_audio:

        st.warning(
            "Please upload a reference voice recording."
        )

        st.stop()

    if not ref_text.strip():

        st.warning(
            "Please provide the transcript of the reference audio."
        )

        st.stop()

    if not text_for_tts.strip():

        st.warning(
            "Please enter text to generate."
        )

        st.stop()

    try:

        with st.status(
            "Generating your voice...",
            expanded=True,
        ) as status:

            st.write("Preparing reference audio...")

            st.write(
                f"Emotion: **{emotion}**"
            )

            st.write(
                f"Speed: **{speed:.2f}x**"
            )

            st.write(
                f"Language: **{language}**"
            )

            st.write("Running OmniVoice inference...")

            audio = generate_audio(
                model=model,
                text=text_for_tts,
                ref_audio=reference_audio_path,
                ref_text=ref_text,
                language=language,
                emotion=emotion,
                speed=speed,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
            )

            status.update(
                label="Generation complete!",
                state="complete",
                expanded=False,
            )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ).name

        sf.write(
            output_path,
            np.asarray(audio).squeeze(),
            24000,
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    🔊 Generated Voice
                </div>

                <div class="card-subtitle">
                    Your cloned voice is ready.
                </div>

            </div>
            """
        )

        st.audio(
            output_path,
            format="audio/wav",
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        with open(output_path, "rb") as f:

            audio_bytes = f.read()

        st.download_button(
            label="⬇️ Download WAV",
            data=audio_bytes,
            file_name="omnivoice_output.wav",
            mime="audio/wav",
            use_container_width=True,
        )


        # ----------------------------------------------------
        # GENERATION INFO
        # ----------------------------------------------------

        duration = len(audio) / 24000

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Duration",
                f"{duration:.1f}s",
            )

        with c2:

            st.metric(
                "Emotion",
                emotion,
            )

        with c3:

            st.metric(
                "Speed",
                f"{speed:.2f}x",
            )


    except Exception as e:

        st.error(
            "Voice generation failed."
        )

        with st.expander("Show technical error"):

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Powered by OmniVoice · Voice AI Studio
    </div>
    """,
    unsafe_allow_html=True,
)