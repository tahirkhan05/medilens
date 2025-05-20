import streamlit as st
import torch
import os
import io
import base64
import transformers    
from PIL import Image
from typing import Optional, Union, List

# More aggressive workaround for the file watcher issue
os.environ["STREAMLIT_WATCHDOG_EXTENSIONS"] = ""

# Try to disable the problematic file watching
try:
    # Patch torch._classes to avoid the file watcher error
    import sys
    import types
    
    # Create a dummy module with empty __path__
    dummy_module = types.ModuleType("dummy")
    dummy_module.__path__ = []
    
    # Patch torch._classes.__path__ with the dummy module
    if "_classes" in sys.modules.get("torch", {}).__dict__:
        sys.modules["torch"]._classes.__path__ = dummy_module.__path__
except Exception as e:
    st.sidebar.warning(f"Module patching failed, but app may still work: {e}")

# First try to patch DynamicCache before importing transformers
try:
    # Save the original import
    original_import = __import__
    
    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        
        # Add monkey patch to DynamicCache if it's being imported
        if name == 'transformers.cache_utils' and hasattr(module, 'DynamicCache'):
            if not hasattr(module.DynamicCache, 'get_max_length'):
                def get_max_length(self):
                    if hasattr(self, 'get_max_cache_shape'):
                        return self.get_max_cache_shape()[1]
                    else:
                        return 2048
                module.DynamicCache.get_max_length = get_max_length
        
        return module
    
    # Replace the built-in import with our patched version
    __builtins__['__import__'] = patched_import
    
except Exception as patch_error:
    print(f"Failed to patch imports: {patch_error}")

# Now proceed with importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as import_error:
    st.error(f"Failed to import transformers: {import_error}")
    st.warning("Try installing version 4.36.0: pip install transformers==4.36.0")
    
# Set page config
st.set_page_config(
    page_title="BiMediX2-4B Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on every interaction
def load_model_and_tokenizer(model_path: str):
    """Load the BiMediX2-4B model and tokenizer."""
    try:
        # Load tokenizer with trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Apply monkey patch to handle DynamicCache issues
        try:
            from transformers.cache_utils import DynamicCache
            
            # Add the missing method if it doesn't exist
            if not hasattr(DynamicCache, 'get_max_length'):
                def get_max_length(self):
                    if hasattr(self, 'get_max_cache_shape'):
                        return self.get_max_cache_shape()[1]
                    else:
                        # Fallback to a reasonable default
                        return 2048
                
                # Apply the monkey patch
                DynamicCache.get_max_length = get_max_length
                st.success("✅ Applied compatibility patch for DynamicCache")
        except Exception as patch_error:
            st.warning(f"Cache compatibility patch failed: {patch_error}")
        
        # Try loading model using older method first (for transformers < 4.36)
        try:
            # Load model with appropriate settings and trust_remote_code=True
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception as e1:
            st.warning(f"Modern model loading failed: {e1}. Trying legacy method...")
            # Try fallback method for older versions
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_cache=False,  # Disable KV cache to avoid the error
                    trust_remote_code=True
                )
            except Exception as e2:
                raise Exception(f"Failed to load model with multiple methods: {e1} | {e2}")
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        
        # More specific error handling
        if "trust_remote_code" in str(e):
            st.warning("This model requires setting trust_remote_code=True. The code has been updated to include this parameter.")
        elif "The checkpoint you are trying to load" in str(e):
            st.warning("Try updating transformers with: pip install --upgrade transformers")
            st.warning("Or install the latest version: pip install git+https://github.com/huggingface/transformers.git")
        elif "CUDA out of memory" in str(e):
            st.warning("GPU memory insufficient. Try running on CPU or with a smaller model.")
        
        return None, None

def process_image(image):
    """Process uploaded image for model input."""
    # Resize image if needed
    max_size = 512
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Convert to RGB if in another mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_response(model, tokenizer, prompt: str, image: Optional[Image.Image] = None, 
                     max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """Generate response from the model with text and/or image input."""
    try:
        # Handle input with or without image
        if image is not None:
            processed_image = process_image(image)
            enhanced_prompt = f"[Image input provided] {prompt}"
            inputs = tokenizer.encode(enhanced_prompt, return_tensors="pt")
        else:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Apply monkey patch to handle DynamicCache issues
        # This addresses the 'get_max_length' deprecated method
        try:
            import transformers
            from transformers.cache_utils import DynamicCache
            
            # Add the missing method if it doesn't exist
            if not hasattr(DynamicCache, 'get_max_length'):
                def get_max_length(self):
                    if hasattr(self, 'get_max_cache_shape'):
                        return self.get_max_cache_shape()[1]
                    else:
                        # Fallback to a reasonable default
                        return 2048
                
                # Apply the monkey patch
                DynamicCache.get_max_length = get_max_length
        except Exception as patch_error:
            st.warning(f"Cache compatibility patch failed: {patch_error}")
        
        # Generate response with minimal parameters to avoid cache issues
        with torch.no_grad():
            try:
                # First attempt: using generation_config
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    use_cache=True  # Explicitly enable caching
                )
            except Exception as gen_error:
                # If that fails, try the simplest possible generation
                st.warning(f"Advanced generation failed: {gen_error}. Trying simplified generation.")
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=False  # Disable caching entirely
                )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    # Title and description
    st.title("🏥 BiMediX2-4B Medical AI Assistant")
    st.markdown("---")
    st.markdown("This is a medical AI assistant based on the BiMediX2-4B model. Please consult with healthcare professionals for medical advice.")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
                
        # Model path input with a more user-friendly default
        model_path = st.text_input(
            "Model Path/Name",
            value="MBZUAI/BiMediX2-4B",  # Changed to a more likely HF path
            help="Path to the BiMediX2-4B repository or Hugging Face model name"
        )
        
        # Input type selection
        st.subheader("Input Options")
        input_type = st.radio(
            "Select input type:",
            options=["Text Only", "Image Only", "Both Text and Image"],
            index=0
        )
        
        # Generation parameters
        st.subheader("Generation Parameters")
        max_length = st.slider("Max New Tokens", 100, 2044, 512, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        
        # Advanced options
        st.subheader("Advanced Options")
        if 'disable_cache' not in st.session_state:
            st.session_state.disable_cache = False
        st.session_state.disable_cache = st.checkbox("Disable KV Cache", value=st.session_state.disable_cache, 
                                   help="Check this if you're experiencing DynamicCache errors")
        
        # Model info
        st.subheader("📊 Model Information")
        if torch.cuda.is_available():
            st.success(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        else:
            st.info("💻 Running on CPU")
            
        # Display transformers version
        try:
            st.text(f"Transformers version: {transformers.__version__}")
            # Check if we're using a version that might have the issue
            if hasattr(transformers, "__version__"):
                version = transformers.__version__.split(".")
                if int(version[0]) >= 4 and int(version[1]) >= 48:
                    st.warning("You're using transformers ≥ v4.48 which has breaking changes in the cache system. Consider downgrading to v4.36.0.")
        except:
            st.text("Transformers version: Unknown")
    
    # Main content area
    # Load model
    if st.button("🔄 Load Model", type="primary"):
        with st.spinner("Loading BiMediX2-4B model..."):
            model, tokenizer = load_model_and_tokenizer(model_path)
            if model and tokenizer:
                st.success("✅ Model loaded successfully!")
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
    
    # Chat interface
    st.subheader("💬 Chat with BiMediX2-4B")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message and message["image"] is not None:
                st.image(message["image"], caption="Uploaded Image", use_column_width=True)
            st.markdown(message["content"])
    
    # Input area based on selected input type
    user_input = ""
    user_image = None
    
    if input_type in ["Text Only", "Both Text and Image"]:
        user_input = st.text_area("Enter your medical question:", height=100, key="text_input")
    
    if input_type in ["Image Only", "Both Text and Image"]:
        uploaded_file = st.file_uploader("Upload a medical image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                user_image = Image.open(uploaded_file)
                st.image(user_image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Submit button
    if st.button("Submit", key="submit_button"):
        if (input_type == "Text Only" and user_input) or \
           (input_type == "Image Only" and user_image is not None) or \
           (input_type == "Both Text and Image" and user_input and user_image is not None):
            
            # Add user message to chat history
            user_message = {
                "role": "user", 
                "content": user_input if user_input else "Image analysis request",
                "image": user_image
            }
            st.session_state.messages.append(user_message)
            
            # Generate response if model is loaded
            if hasattr(st.session_state, 'model') and hasattr(st.session_state, 'tokenizer'):
                with st.spinner("Generating response..."):
                    # Check if we should disable cache
                    use_cache = not (st.session_state.get('disable_cache', False))
                    
                    # Try generation with fallback options
                    try:
                        response = generate_response(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            user_input,
                            user_image,
                            max_length,
                            temperature,
                            top_p
                        )
                    except Exception as gen_error:
                        st.error(f"Error during generation: {gen_error}")
                        # Fallback to simplest possible generation
                        try:
                            st.warning("Attempting simplified generation...")
                            inputs = st.session_state.tokenizer.encode(user_input, return_tensors="pt")
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                                st.session_state.model = st.session_state.model.to("cuda")
                            
                            with torch.no_grad():
                                outputs = st.session_state.model.generate(
                                    inputs,
                                    max_new_tokens=max_length,
                                    use_cache=False
                                )
                            
                            response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            # Remove the input prompt from the response
                            if user_input in response:
                                response = response.replace(user_input, "").strip()
                        except Exception as fallback_error:
                            response = f"Generation failed: {fallback_error}. Try installing transformers v4.36.0: `pip install transformers==4.36.0`"
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "image": None
                    })
                    
                    # Force refresh to display new messages
                    st.rerun()
            else:
                st.error("Please load the model first!")
        else:
            st.warning("Please provide the required input based on your selected input type.")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>BiMediX2-4B Medical AI Assistant | Built with Streamlit</p>
            <p>⚠️ For educational and research purposes only</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
