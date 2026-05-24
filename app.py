import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from pathlib import Path
import gdown
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WildEye",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark forest theme */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #3fb950;
}

.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.tag {
    display: inline-block;
    background: #1f2d1f;
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}

.warning-tag {
    background: #2d1f1f;
    color: #f85149;
    border-color: #f85149;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3fb950 0%, #79c0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 2rem;
}

.result-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
}

.class-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

.sidebar-section {
    background: #161b22;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    border: 1px solid #21262d;
}

div[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

.stSlider > label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #8b949e !important;
}

.upload-hint {
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

CLASS_EMOJI = {
    'butterfly': '🦋', 'cat': '🐱', 'chicken': '🐔', 'cow': '🐄',
    'dog': '🐶', 'elephant': '🐘', 'horse': '🐎', 'sheep': '🐑',
    'spider': '🕷️', 'squirrel': '🐿️',
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the domain-augmented model. Tries local path first, then HF Hub."""
    num_classes = len(CLASSES)
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    # Try local weights first (for Colab / local dev)
    local_paths = [
        Path('models/domain_aug_best.pth'),
        Path('/content/drive/MyDrive/deeplearn/hack1/models/domain_aug_best.pth'),
    ]
    for p in local_paths:
        if p.exists():
            ckpt = torch.load(p, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            return model, str(p)

    return model, 'pretrained_only'


# ── Augmentation builder ──────────────────────────────────────────────────────
def build_transform(night_ir, motion_blur, low_light, noise, occlusion):
    """Build a transform from slider values (all 0-1 floats)."""
    ops = [A.Resize(IMAGE_SIZE, IMAGE_SIZE)]

    if night_ir > 0:
        ops.append(A.ToGray(p=1.0))
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=(-night_ir * 0.5, -night_ir * 0.1),
            contrast_limit=(-0.1, 0.1), p=1.0
        ))

    if motion_blur > 0:
        blur_limit = max(3, int(motion_blur * 20))
        ops.append(A.MotionBlur(blur_limit=(blur_limit, blur_limit + 4), p=1.0))

    if low_light > 0:
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=(-low_light * 0.6, -low_light * 0.2),
            contrast_limit=(-low_light * 0.3, 0.0), p=1.0
        ))

    if noise > 0:
        ops.append(A.GaussNoise(std_range=(noise * 0.05, noise * 0.15), p=1.0))

    if occlusion > 0:
        n_holes = max(1, int(occlusion * 10))
        hole_size = max(10, int(occlusion * 50))
        ops.append(A.CoarseDropout(
            num_holes_range=(n_holes, n_holes + 2),
            hole_height_range=(hole_size, hole_size + 10),
            hole_width_range=(hole_size, hole_size + 10),
            p=1.0
        ))

    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def predict(model, img_np, transform):
    tensor = transform(image=img_np)['image'].unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    top_idx  = int(probs.argmax())
    return CLASSES[top_idx], probs


def denormalize(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


# ── UI ────────────────────────────────────────────────────────────────────────
# Hero
st.markdown('<div class="hero-title">🦌 WildEye</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Robust Wildlife Classifier · Transfer Learning + Domain Augmentation</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("### 🔬 Robustness Lab")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.8rem;font-family:Space Mono,monospace;"
        "margin-bottom:16px'>Simulate real-world camera-trap conditions. "
        "Watch how each model handles them.</div>",
        unsafe_allow_html=True
    )

    night_ir     = st.slider("🌙 Night / IR",      0.0, 1.0, 0.0, 0.05)
    motion_blur  = st.slider("💨 Motion Blur",     0.0, 1.0, 0.0, 0.05)
    low_light    = st.slider("🔅 Low Light",       0.0, 1.0, 0.0, 0.05)
    noise        = st.slider("📡 Sensor Noise",    0.0, 1.0, 0.0, 0.05)
    occlusion    = st.slider("🌿 Occlusion",       0.0, 1.0, 0.0, 0.05)

    if st.button("⚡ Worst-case scenario", use_container_width=True):
        st.session_state['preset'] = 'worst'
        st.rerun()

    if st.button("✨ Reset to clean",  use_container_width=True):
        st.session_state['preset'] = 'clean'
        st.rerun()

    if st.session_state.get('preset') == 'worst':
        night_ir = motion_blur = low_light = noise = occlusion = 0.8
    elif st.session_state.get('preset') == 'clean':
        night_ir = motion_blur = low_light = noise = occlusion = 0.0

    st.divider()
    st.markdown("### 📊 Project Results")
    st.markdown("""
    <div class='metric-card' style='margin-bottom:8px'>
        <div class='metric-value'>+19.3%</div>
        <div class='metric-label'>Robustness gain<br>worst-case conditions</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>-1.1%</div>
        <div class='metric-label'>Cost on clean data</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem;font-family:Space Mono,monospace'>"
        "Model: EfficientNet-B0<br>"
        "Training: Animals-10 (8,695 imgs)<br>"
        "Augmentations: 6 domain-specific<br>"
        "Mini Hackathon #1 — WildEye</div>",
        unsafe_allow_html=True
    )

# Main content
col_upload, col_results = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload an image")
    uploaded = st.file_uploader(
        "Camera trap image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if not uploaded:
        st.markdown(
            "<div class='upload-hint'>📷 Drop an animal photo here<br>"
            "<span style='font-size:0.7rem'>Supports: jpg · png · webp</span></div>",
            unsafe_allow_html=True
        )
    else:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np  = np.array(img_pil)

        tf = build_transform(night_ir, motion_blur, low_light, noise, occlusion)

        import io
        from PIL import Image as PILImage
        aug_tensor = tf(image=img_np)['image']
        aug_display = (denormalize(aug_tensor.unsqueeze(0)) * 255).astype(np.uint8)

        any_aug = any([night_ir, motion_blur, low_light, noise, occlusion])

        if any_aug:
            c1, c2 = st.columns(2)
            c1.image(img_pil, caption="Original", use_container_width=True)
            c2.image(aug_display, caption="As model sees it", use_container_width=True)
        else:
            st.image(img_pil, caption="Original (no perturbations)", use_container_width=True)

        # Active perturbation tags
        active = []
        if night_ir    > 0: active.append(f"🌙 Night IR ({night_ir:.0%})")
        if motion_blur > 0: active.append(f"💨 Blur ({motion_blur:.0%})")
        if low_light   > 0: active.append(f"🔅 Low Light ({low_light:.0%})")
        if noise       > 0: active.append(f"📡 Noise ({noise:.0%})")
        if occlusion   > 0: active.append(f"🌿 Occlusion ({occlusion:.0%})")

        if active:
            tags_html = " ".join(f"<span class='tag'>{t}</span>" for t in active)
            st.markdown(tags_html, unsafe_allow_html=True)

with col_results:
    if uploaded:
        st.markdown("#### Prediction")
        model, source = load_model()
        pred_class, probs = predict(model, img_np, tf)

        conf = probs.max()
        emoji = CLASS_EMOJI.get(pred_class, "🐾")

        # Main prediction card
        color = "#3fb950" if conf > 0.7 else "#d29922" if conf > 0.4 else "#f85149"
        st.markdown(f"""
        <div class='result-box'>
            <div style='font-size:3rem;margin-bottom:8px'>{emoji}</div>
            <div style='font-size:2rem;font-weight:800;color:{color}'>{pred_class.upper()}</div>
            <div style='font-family:Space Mono,monospace;color:#8b949e;font-size:0.85rem;margin-top:4px'>
                Confidence: <span style='color:{color};font-weight:700'>{conf:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-5 bar chart
        st.markdown("<br>**Top predictions**", unsafe_allow_html=True)
        top5_idx  = probs.argsort()[::-1][:5]
        top5_cls  = [CLASSES[i] for i in top5_idx]
        top5_prob = [probs[i] for i in top5_idx]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.8))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        colors_bar = ['#3fb950' if c == pred_class else '#30363d' for c in top5_cls]
        bars = ax.barh(
            [f"{CLASS_EMOJI.get(c,'🐾')} {c}" for c in top5_cls[::-1]],
            top5_prob[::-1],
            color=colors_bar[::-1], edgecolor='none', height=0.6
        )
        for bar, val in zip(bars, top5_prob[::-1]):
            ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', color='#e6edf3',
                    fontsize=8, fontfamily='monospace')
        ax.set_xlim(0, 1.05)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.spines[['top','right','bottom']].set_visible(False)
        ax.spines['left'].set_color('#30363d')
        ax.xaxis.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Robustness warning
        if any_aug:
            severity = np.mean([night_ir, motion_blur, low_light, noise, occlusion])
            if conf < 0.5:
                st.markdown(
                    "<div style='background:#2d1f1f;border:1px solid #f85149;border-radius:8px;"
                    "padding:12px;font-family:Space Mono,monospace;font-size:0.8rem;color:#f85149;"
                    "margin-top:8px'>⚠️ Low confidence under these conditions — "
                    "this is exactly what domain augmentation trains against.</div>",
                    unsafe_allow_html=True
                )
            elif conf > 0.8:
                st.markdown(
                    "<div style='background:#1f2d1f;border:1px solid #3fb950;border-radius:8px;"
                    "padding:12px;font-family:Space Mono,monospace;font-size:0.8rem;color:#3fb950;"
                    "margin-top:8px'>✓ High confidence maintained despite perturbations — "
                    "domain augmentation working as intended.</div>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<div style='color:#8b949e;font-family:Space Mono,monospace;font-size:0.85rem;"
            "padding:40px 0;text-align:center'>"
            "Upload an image to see predictions.<br><br>"
            "Use the sidebar sliders to simulate<br>real camera-trap conditions.</div>",
            unsafe_allow_html=True
        )

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-family:Space Mono,monospace;"
    "font-size:0.75rem'>WildEye · Mini Hackathon #1 · "
    "EfficientNet-B0 + Domain Augmentation · "
    "Robustness gain: +19.3% under worst-case conditions</div>",
    unsafe_allow_html=True
)
