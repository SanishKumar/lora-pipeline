import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# Configure page
st.set_page_config(
    page_title="LoRA Fine-Tuner UI",
    page_icon="🎨",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint, data=None, method="GET"):
    """Make API calls to the backend"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    st.title("🎨 LoRA Fine-Tuner & Image Generator")
    st.markdown("Generate custom artwork using fine-tuned Stable Diffusion models")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Generation Settings")
        
        # Check API health
        health = call_api("/health")
        if health:
            st.success(f"✅ API Connected - Device: {health.get('device', 'unknown')}")
        else:
            st.error("❌ API Disconnected")
            return

        # Model information
        models = call_api("/models")
        if models:
            st.info("**Current Models:**")
            st.text(f"Base: {models.get('base_model', 'N/A')}")
            st.text(f"LoRA: {models.get('lora_model', 'N/A')}")

        st.divider()

        # Generation parameters
        use_lora = st.checkbox("Use LoRA Model", value=True)
        lora_scale = st.slider("LoRA Scale", 0.0, 2.0, 1.0, 0.1) if use_lora else 1.0
        
        num_steps = st.slider("Inference Steps", 10, 100, 50)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=0)
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=0)
        
        seed = st.number_input("Seed (0 for random)", 0, 2**31, 0)
        if seed == 0:
            seed = None

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Prompt Input")
        
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="a beautiful landscape painting in impressionist style",
            height=100
        )
        
        negative_prompt = st.text_area(
            "Negative prompt (optional):",
            placeholder="blurry, low quality, distorted",
            height=60
        )
        
        if st.button("🎨 Generate Image", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("Please enter a prompt!")
                return
            
            # Prepare request
            request_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "use_lora": use_lora,
                "lora_scale": lora_scale
            }
            
            # Show loading spinner
            with st.spinner("Generating image... This may take a few moments."):
                result = call_api("/generate", request_data, "POST")
            
            if result:
                # Store result in session state
                st.session_state['last_result'] = result
                st.session_state['last_prompt'] = prompt

    with col2:
        st.header("Generated Image")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Decode and display image
            try:
                image_data = base64.b64decode(result['image'])
                image = Image.open(io.BytesIO(image_data))
                
                st.image(image, caption=f"Generated in {result['generation_time']:.2f}s")
                
                # Image info
                st.info(f"**Seed:** {result['seed']}")
                if 'last_prompt' in st.session_state:
                    st.info(f"**Prompt:** {st.session_state['last_prompt']}")
                
                # Download button
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                
                st.download_button(
                    label="📥 Download Image",
                    data=img_buffer.getvalue(),
                    file_name=f"generated_image_{result['seed']}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        else:
            st.info("👈 Enter a prompt and click 'Generate Image' to see results here!")

    # Footer with example prompts
    st.divider()
    st.header("Example Prompts")
    
    examples = [
        "a majestic mountain landscape in watercolor style",
        "a cute cat sitting in a garden, oil painting",
        "futuristic cityscape at sunset, digital art",
        "portrait of a wise old wizard, fantasy art style",
        "abstract geometric patterns in vibrant colors"
    ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.session_state['example_prompt'] = example
                st.rerun()

    # Auto-fill example prompt
    if 'example_prompt' in st.session_state:
        prompt = st.session_state['example_prompt']
        del st.session_state['example_prompt']

if __name__ == "__main__":
    main()