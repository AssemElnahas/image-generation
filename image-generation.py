import streamlit as st
from huggingface_hub import InferenceClient
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Echo - Text to Image",
    page_icon="❤️",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# ---------------- CUSTOM CSS (Matching Your Image Style) ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #46657d 0%, #8b949c 100%);
        color: white;
    }

    .main-container {
        text-align: center;
        padding-top: 20px;
        max-width: 520px;
        margin: 0 auto;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 20px;
        font-size: 20px;
        font-weight: 500;
    }

    .logo-container {
        margin: 25px 0 15px 0;
    }

    .title {
        font-size: 28px;
        font-weight: 500;
        color: white;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 16px;
        color: white;
        margin-bottom: 35px;
        opacity: 0.95;
    }

    /* Prompt Box */
    .stTextArea textarea {
        background-color: #2a2a2a !important;
        color: white !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 16px;
        padding: 16px !important;
        min-height: 58px !important;
    }

    .stTextArea label {
        display: none !important;
    }

    /* Generate Button */
    .stButton>button {
        width: 100%;
        background-color: #2f2f2f;
        color: white;
        border-radius: 15px;
        font-size: 17px;
        padding: 12px 0;
        border: none;
        margin-top: 10px;
    }

    /* Download Button */
    .stDownloadButton>button {
        width: 100%;
        background-color: #1f8a4e;
        color: white;
        border-radius: 15px;
        font-size: 17px;
        padding: 12px 0;
        border: none;
        margin-top: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2a2a2a;
        color: white;
    }

    [data-testid="stSidebar"] .stTextArea textarea {
        background-color: #1f1f1f !important;
        color: white !important;
        border-radius: 12px;
    }

    .arabic {
        text-align: right;
        color: #aaaaaa;
        font-size: 14px;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR - API Key ----------------
with st.sidebar:
    st.header("🔑 Settings")
    st.markdown("### Hugging Face API Key")
    
    api_key = st.text_area(
        label="Paste your HF Token",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        height=120,
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    
    st.info("Using **nscale** provider with FLUX.1-schnell (fast generation).")

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <span style="font-weight:500;">Image Generation</span>
</div>
""", unsafe_allow_html=True)

# Logo
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("logo.png", width=180)
st.markdown('</div>', unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="title">Image Generation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Here to help you with anything you need</div>',
    unsafe_allow_html=True
)

# Prompt Input
prompt = st.text_area(
    label="Ask Echo ?",
    placeholder="Describe the image you want to generate...",
    height=100,
    label_visibility="collapsed"
)

# Generate Button
if st.button("Generate Image"):
    if not api_key or api_key.strip() == "":
        st.error("⚠️ Please enter your Hugging Face API Key in the sidebar first.")
    elif not prompt or prompt.strip() == "":
        st.warning("Please enter a prompt to generate an image.")
    else:
        try:
            with st.spinner("Generating image..."):
                client = InferenceClient(
                    provider="nscale",
                    api_key=api_key.strip(),
                )

                image = client.text_to_image(
                    prompt=prompt,
                    model="black-forest-labs/FLUX.1-schnell",
                )

                # Save to session state so it persists
                st.session_state.generated_image = image
                
                st.success("✅ Image generated successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Tips:\n• Make sure your API key is valid\n• Check if you have enough credits on nscale provider")

# ==================== DISPLAY GENERATED IMAGE + DOWNLOAD ====================
if st.session_state.generated_image is not None:
    st.image(
        st.session_state.generated_image,
        caption="Generated Image",
        use_container_width=True
    )
    
    # Convert PIL image to bytes for download
    buf = io.BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    st.download_button(
        label="📥 Download Image as PNG",
        data=byte_data,
        file_name="echo_generated_image.png",
        mime="image/png",
        use_container_width=True
    )

st.markdown('<div class="arabic">؟إبداع</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)