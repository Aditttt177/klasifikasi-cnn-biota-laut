import streamlit as st, numpy as np, pandas as pd, json, glob
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Klasifikasi Biota Laut CNN", layout="wide")

@st.cache_resource
def muat():
    return tf.keras.models.load_model("model_cnn.keras")
model = muat()
R = json.load(open("ringkasan.json"))
kelas = R["kelas"]

st.title("Klasifikasi Biota Laut dengan CNN")
st.write("Dashboard klasifikasi citra biota laut (Oceanic Life Dataset, Kaggle) "
         "memakai CNN. Label dibuat otomatis dengan CLIP (pseudo-label).")

st.header("Ringkasan Dataset")
a, b, c, d = st.columns(4)
a.metric("Jumlah gambar", R["total"]); b.metric("Jumlah kelas", len(kelas))
c.metric("Data latih", R["train"]); d.metric("Data uji", R["test"])

st.header("Distribusi Data per Kelas")
st.image("eda_distribusi_kelas.png")
st.caption("Fish mendominasi (2180) sedangkan seaweed hanya 128; rasio sekitar 17:1 (class imbalance), "
           "ditangani dengan class weight.")

st.header("Hasil Pelatihan")
st.image("hasil_kurva_training.png")
st.caption("Akurasi train dan uji naik dan konvergen tanpa overfitting parah; "
           "fluktuasi validasi karena data uji kecil dan label pseudo bernoise.")

st.header("Confusion Matrix")
st.image("hasil_confusion_matrix.png")
st.caption("Diagonal dominan; kesalahan terbesar fish diprediksi coral karena latar terumbu karang.")

st.header("Evaluasi Model")
st.dataframe(pd.read_csv("metrik_per_kelas.csv", index_col=0))
st.dataframe(pd.read_csv("metrik_cnn.csv", index_col=0))

st.header("Contoh Prediksi dan Grad-CAM")
x, y = st.columns(2)
x.image("hasil_prediksi_contoh.png", caption="Prediksi vs label asli")
y.image("hasil_gradcam.png", caption="Grad-CAM")

st.header("Prediksi Gambar Baru (real-time)")
f = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
if f:
    im = Image.open(f).convert("RGB")
    st.image(im, width=300)
    x = np.asarray(im.resize((48, 48)), dtype="float32")[None] / 255.0
    p = model.predict(x, verbose=0)[0]
    st.success(f"Prediksi: {kelas[int(p.argmax())]} ({p.max():.1%})")
    st.bar_chart(pd.Series(p, index=kelas))
