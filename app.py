import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(
    page_title="PneumoNet — AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #f0f4f8; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #e8f0fe 0%, #e3f2fd 50%, #e8f5e9 100%);
        border: 1px solid #dbe8fc;
        border-radius: 24px;
        padding: 2.8rem 2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.3rem 1rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        color: #1e3a5f;
        margin-bottom: 0.8rem;
        letter-spacing: -0.01em;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 0.98rem;
        max-width: 460px;
        margin: 0 auto 2rem;
        line-height: 1.7;
        font-weight: 300;
    }

    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #c7d9f5;
        flex-wrap: wrap;
    }

    .stat-item { text-align: center; }

    .stat-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1d4ed8;
    }

    .stat-label {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.15rem;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .card.pneumonia {
        border-color: #fecaca;
        background: linear-gradient(135deg, #fff5f5, #fff8f8);
    }

    .card.normal {
        border-color: #bbf7d0;
        background: linear-gradient(135deg, #f0fdf4, #f7fef9);
    }

    .card.inconclusive {
        border-color: #fde68a;
        background: linear-gradient(135deg, #fffbeb, #fffef5);
    }

    .result-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .result-value.pneumonia { color: #e53e3e; }
    .result-value.normal { color: #38a169; }
    .result-value.inconclusive { color: #d97706; }

    .conf-bar-bg {
        background: #edf2f7;
        border-radius: 100px;
        height: 5px;
        margin-top: 0.8rem;
        overflow: hidden;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 100px;
    }

    .section-header {
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1.5px solid #e2e8f0;
    }

    .prob-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .prob-value-normal {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38a169;
    }

    .prob-value-pneumonia {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e53e3e;
    }

    .alert-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        color: #92400e;
        font-size: 0.82rem;
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    .alert-danger {
        background: #fff5f5;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        color: #9b2c2c;
        font-size: 0.82rem;
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    .alert-success {
        background: #f0fff4;
        border: 1px solid #c6f6d5;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        color: #276749;
        font-size: 0.82rem;
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    .empty-state {
        background: white;
        border: 1.5px dashed #cbd5e1;
        border-radius: 18px;
        text-align: center;
        padding: 4rem 2rem;
        color: #94a3b8;
    }

    .disclaimer {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        color: #94a3b8;
        font-size: 0.74rem;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.6;
    }

    .upload-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .gradcam-caption {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">🫁 Medical AI · Deep Learning</div>
    <div class="hero-title">PneumoNet</div>
    <div class="hero-subtitle">
        Upload a chest X-ray for an AI-powered pneumonia assessment
        with visual Grad-CAM explainability.
    </div>
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value">88.1%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">5,216</div>
            <div class="stat-label">Training Images</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">ResNet18</div>
            <div class="stat-label">Architecture</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">Grad-CAM</div>
            <div class="stat-label">Explainability</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('models/pneumonet_best.pth',
                          map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Layout
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📤 Upload X-Ray</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray in JPG or PNG format"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-Ray', use_container_width=True)
        st.caption(f"📁 {uploaded_file.name} · {image.size[0]}×{image.size[1]} px")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0;color:#cbd5e1;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">🩻</div>
            <div style="font-size:0.82rem;">No image uploaded yet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if uploaded_file:
        st.markdown('<div class="section-header">🔬 Diagnosis</div>', unsafe_allow_html=True)

        with st.spinner('Analyzing X-ray...'):
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            classes = ['NORMAL', 'PNEUMONIA']
            prediction = classes[predicted.item()]
            conf_score = confidence.item() * 100
            normal_prob = probabilities[0][0].item() * 100
            pneumonia_prob = probabilities[0][1].item() * 100

            if conf_score < 70:
                card_class = "inconclusive"
                icon = "⚠️"
                display = "INCONCLUSIVE"
                bar_color = "#d97706"
            elif prediction == 'PNEUMONIA':
                card_class = "pneumonia"
                icon = "🔴"
                display = "PNEUMONIA DETECTED"
                bar_color = "#e53e3e"
            else:
                card_class = "normal"
                icon = "🟢"
                display = "NORMAL"
                bar_color = "#38a169"

            st.markdown(f"""
            <div class="card {card_class}">
                <div class="result-label">AI Diagnosis</div>
                <div class="result-value {card_class}">{icon} {display}</div>
                <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.3rem;">
                    Confidence: <strong style="color:#475569">{conf_score:.1f}%</strong>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_score}%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="result-label">Normal</div>
                    <div class="prob-value-normal">{normal_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="result-label">Pneumonia</div>
                    <div class="prob-value-pneumonia">{pneumonia_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            if conf_score < 70:
                st.markdown("""
                <div class="alert-warning">⚠️ Confidence below 70% — result is inconclusive. Please consult a radiologist for a definitive diagnosis.</div>
                """, unsafe_allow_html=True)
            elif prediction == 'PNEUMONIA':
                st.markdown("""
                <div class="alert-danger">🚨 Pneumonia indicators detected. Please consult a qualified medical professional immediately.</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">✅ No pneumonia indicators detected. Regular check-ups are still recommended.</div>
                """, unsafe_allow_html=True)

        # Grad-CAM
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">🧠 Grad-CAM Explainability</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="gradcam-caption">
            🔴 Red/yellow regions = areas the model focused on most to reach its decision.
            Blue regions = less relevant areas.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner('Generating heatmap...'):
            target_layers = [model.layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(predicted.item())]
            )
            rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam[0])

            g1, g2 = st.columns(2)
            g1.image(rgb_img, caption='Original X-Ray', use_container_width=True)
            g2.image(visualization, caption='Grad-CAM Heatmap', use_container_width=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
            <div style="font-weight:500;color:#64748b;margin-bottom:0.4rem;">Awaiting X-Ray</div>
            <div style="font-size:0.82rem;">Upload a chest X-ray on the left to see the AI diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> PneumoNet is an educational AI project and not a certified medical device.
    Results should never replace professional medical diagnosis. Always consult a qualified radiologist or physician.
</div>
""", unsafe_allow_html=True)