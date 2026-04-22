import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF


st.set_page_config(page_title="UNet Segmentation App", layout="wide")

NUM_CLASSES = 23
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS_PATH = "unet_model.pth" # Replace with your actual saved weights path

# ==========================================
# Model Loading (Cached for performance)
# ==========================================
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=NUM_CLASSES,
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        st.sidebar.success("Model weights loaded successfully!")
    except FileNotFoundError:
        st.sidebar.warning(f"Weights '{MODEL_WEIGHTS_PATH}' not found. Using untrained model for demonstration.")
        
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==========================================
# Navigation
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training Metrics", "Model Inference"])

# ==========================================
# Page 1: Training Metrics & Plots
# ==========================================
if page == "Training Metrics":
    st.title("Model Training Phase Metrics")
    
    # 1. Display Test Set Scores
    st.subheader("Test Set Performance")
    col1, col2 = st.columns(2)
    # Replace these hardcoded values with your actual evaluated test scores
    col1.metric("Test mIOU", "0.7845") 
    col2.metric("Test mDice", "0.8612")
    
    st.divider()
    
    # 2. Display Training Plots
    st.subheader("Training Curves")
    st.info("Note: Replace the dummy data arrays below with your actual saved training history to show your real curves.")
    
    # Generating dummy history to make the plots render out-of-the-box
    epochs = range(1, 16)
    dummy_loss = [np.exp(-0.3 * e) + np.random.normal(0, 0.05) for e in epochs]
    dummy_miou = [0.8 - 0.5 * np.exp(-0.2 * e) for e in epochs]
    dummy_mdice = [0.9 - 0.6 * np.exp(-0.25 * e) for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss Plot
    axes[0].plot(epochs, dummy_loss, marker='o', color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].grid(True)
    
    # mIOU Plot
    axes[1].plot(epochs, dummy_miou, marker='s', color='blue')
    axes[1].set_title('Training mIOU')
    axes[1].set_xlabel('Epochs')
    axes[1].grid(True)
    
    # mDice Plot
    axes[2].plot(epochs, dummy_mdice, marker='^', color='green')
    axes[2].set_title('Training mDice')
    axes[2].set_xlabel('Epochs')
    axes[2].grid(True)
    
    st.pyplot(fig)

# ==========================================
# Page 2: Model Inference
# ==========================================
elif page == "Model Inference":
    st.title("Upload Test Images for Segmentation")
    st.write("Upload up to 4 input images and their corresponding ground truth masks to compare against the model's predictions.")
    
    col_img, col_mask = st.columns(2)
    with col_img:
        uploaded_images = st.file_uploader("Upload Test Images (Max 4)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    with col_mask:
        uploaded_masks = st.file_uploader("Upload Ground Truth Masks (Max 4)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_images and uploaded_masks:
        if len(uploaded_images) > 4 or len(uploaded_masks) > 4:
            st.warning("Please upload a maximum of 4 images and 4 masks.")
        elif len(uploaded_images) != len(uploaded_masks):
            st.warning("The number of uploaded images must match the number of uploaded masks.")
        else:
            st.success("Files uploaded successfully! Generating predictions...")
            st.divider()
            
            # Process each image-mask pair
            for idx, (img_file, mask_file) in enumerate(zip(uploaded_images[:4], uploaded_masks[:4])):
                st.subheader(f"Test Sample {idx + 1}")
                
                # Load images
                image = Image.open(img_file).convert("RGB")
                gt_mask = Image.open(mask_file).convert("L") # Assuming masks are grayscale
                
                # Preprocess image for the model
                img_tensor = TF.to_tensor(image).unsqueeze(0).to(DEVICE)
                # Resize if necessary (assuming 256x256 for standard UNet)
                img_tensor = TF.resize(img_tensor, [256, 256])
                
                # Run Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    # Get class predictions (argmax over channel dimension)
                    pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
                
                # Normalize prediction mask for visualization (0-255 scale)
                # Multiplied to make the 23 classes visually distinct
                pred_mask_vis = (pred_mask * (255 / NUM_CLASSES)).astype(np.uint8)
                
                # Display Results side-by-side
                disp_col1, disp_col2, disp_col3 = st.columns(3)
                
                with disp_col1:
                    st.image(image, caption="Original Input Image", use_column_width=True)
                with disp_col2:
                    st.image(gt_mask, caption="Ground Truth Mask", use_column_width=True)
                with disp_col3:
                    st.image(pred_mask_vis, caption="Model Prediction", use_column_width=True, clamp=True)
                
                st.write("---")