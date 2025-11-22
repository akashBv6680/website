# app.py
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io

# Multi-model interface (extensible)
MODEL_OPTIONS = ["Gemini Flash 2.5 (Vision)", "VLM (Vision-Language)", "OpenAI GPT-4V", "Future Model..."]

st.title("AI Animation Creator (Gemini & Vision Models)")

input_text = st.text_area("Enter your script/topic for animation:", "")
selected_model = st.selectbox("Choose Model", MODEL_OPTIONS)
img_upload = st.file_uploader("Upload supporting image (optional for VLM/Gemini Vision)", type=["jpg", "png", "jpeg"])

# Get API key from Streamlit secrets
api_key = st.secrets.get("GEMINI_API_KEY") 

# --- Gemini 2.5 Flash Support Function ---
def call_gemini_flash_2_5(text: str, image_file: st.runtime.uploaded_file_manager.UploadedFile):
    """
    Calls the Gemini 2.5 Flash model using the Google GenAI SDK.
    It prepares the content for multimodal input (text + image).
    """
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please set it.")
        return "Error: API key missing."
    
    try:
        # Initialize the client with the API key
        client = genai.Client(api_key=api_key)
        
        # Prepare the parts list for the model's contents
        parts = [text]

        if image_file is not None:
            # Read the uploaded image file into a PIL Image object
            image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            parts.insert(0, image) # Insert image before text for better context

        # Use the gemini-2.5-flash model for vision/multimodal tasks
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parts,
            config=types.GenerateContentConfig(
                # You can add generation config here, e.g.,
                # temperature=0.7, 
                # max_output_tokens=2048 
            )
        )
        
        return response.text
    except Exception as e:
        st.error(f"An error occurred during the Gemini API call: {e}")
        return "Gemini API call failed."

def call_vlm(text, image_file):
    # Example placeholder for VLM (e.g. Hugging Face Vision Language Model)
    return "VLM Output: [Simulated]"

# Main model switch
if st.button("Generate Animation Script"):
    if not input_text.strip():
        st.warning("Please enter a script or topic for the animation.")
    else:
        if selected_model == "Gemini Flash 2.5 (Vision)":
            st.info("Calling Gemini 2.5 Flash...")
            result = call_gemini_flash_2_5(input_text, img_upload)
            st.success("Generation Complete!")
            st.markdown("### Animation Script Result")
            st.write(result)
        elif selected_model == "VLM (Vision-Language)":
            result = call_vlm(input_text, img_upload)
            st.success(result)
        elif selected_model == "OpenAI GPT-4V":
            st.warning("Add GPT-4V API logic here.")
        else:
            st.warning("Model integration in progress...")

st.info("You can extend this app by adding other models and customizing each model function.")
