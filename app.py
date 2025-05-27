import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set page configuration
st.set_page_config(page_title="IKMCluster Surabaya", layout="wide")

# Custom styling for sidebar and layout
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f5f7fa;
        padding: 2rem 1rem;
    }
    .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .css-1v0mbdj.etr89bj1 {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background-color: #0077b6;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #023e8a;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("IKMCluster Surabaya")
menu = st.sidebar.radio("Navigasi", ["About", "Upload Data", "Preprocessing", "Clustering (BIRCH)", "Evaluasi", "Download"])

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None

if menu == "About":
    st.title("📌 Tentang Aplikasi")
    st.markdown("""
    **IKMCluster Surabaya** adalah aplikasi analitik interaktif untuk klasterisasi data Industri Kecil Menengah (IKM) Kota Surabaya menggunakan algoritma **BIRCH**. Aplikasi ini dirancang untuk membantu analisis kelompok usaha berdasarkan data karakteristik modal awal, luas tanah, jumlah tenaga kerja, skala usaha, risiko usaha, dan jenis perusahaan.

    Fitur utama:
    - Upload dan preprocessing data
    - Klasterisasi berbasis algoritma BIRCH
    - Visualisasi hasil klaster
    - Evaluasi kualitas klasterisasi
    - Interpretasi otomatis hasil klaster
    """)

elif menu == "Upload Data":
    st.title("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success("✅ Data berhasil diunggah!")
        st.dataframe(df.head())

elif menu == "Preprocessing":
    st.title("🧹 Preprocessing Data")
    if st.session_state.data is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu.")
    else:
        df = st.session_state.data.copy()

        st.subheader("📊 Statistik Deskriptif")
        st.dataframe(df.describe())

        st.subheader("🩺 Cek Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
        if missing.sum() > 0:
            if st.button("🧹 Hapus Baris dengan Nilai Kosong"):
                df.dropna(inplace=True)
                st.success("✅ Missing values berhasil dihapus.")

        st.subheader("🔠 Encoding Variabel Kategorik")
        try:
            scale_mapping = {'Usaha Mikro': 0, 'Usaha Kecil': 1, 'Usaha Menengah': 2}
            risk_mapping = {'Rendah': 0, 'Menengah Rendah': 1, 'Menengah Tinggi': 2, 'Tinggi': 3}
            company_mapping = {
                'Perorangan': 0, 'Perseroan Terbatas (PT)': 1, 'Persekutuan Komanditer (CV)': 2,
                'PT Perorangan': 3, 'Badan Hukum Lainnya': 4, 'Koperasi': 5, 'Persekutuan Firma (Fa)': 6,
                'Persekutuan dan Perkumpulan': 7, 'Yayasan': 8, 'Perusahaan Umum Daerah (Perumda)': 9,
                'Badan Layanan Umum (BLU)': 10
            }
            df['Skala Usaha'] = df['Skala Usaha'].map(scale_mapping)
            df['Risiko Usaha'] = df['Risiko Usaha'].map(risk_mapping)
            df['Jenis Perusahaan'] = df['Jenis Perusahaan'].map(company_mapping)
            st.success("✅ Encoding berhasil dilakukan.")
        except:
            st.error("❌ Gagal melakukan encoding. Periksa nama dan isi kolom.")

        st.subheader("📏 Normalisasi Variabel Numerik")
        num_cols = ['Jumlah Tenaga Kerja', 'Modal Awal', 'Luas Tanah']
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.success("✅ StandardScaler berhasil diterapkan.")
        st.dataframe(df.head())

        st.session_state.data = df

elif menu == "Clustering (BIRCH)":
    st.title("🌿 BIRCH Clustering")
    if st.session_state.data is None:
        st.warning("⚠️ Silakan preprocessing data terlebih dahulu.")
    else:
        df = st.session_state.data.copy()
        st.subheader("⚙️ Parameter Clustering")
        threshold = st.slider("Threshold", 0.1, 2.0, 0.5, step=0.1)
        branching_factor = st.slider("Branching Factor", 10, 100, 50, step=10)

        if st.button("🚀 Jalankan Clustering"):
            birch = Birch(threshold=threshold, branching_factor=branching_factor)
            cluster_labels = birch.fit_predict(df)
            df['Cluster'] = cluster_labels
            st.session_state.clustered_data = df

            st.success(f"✅ Clustering selesai! Jumlah klaster terbentuk: {df['Cluster'].nunique()}")
            st.dataframe(df.head())

            st.subheader("🔍 Visualisasi Klaster (PCA 2D)")
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(df.drop(columns=["Cluster"]))
            df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
            df_plot['Cluster'] = cluster_labels
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
            plt.title("Visualisasi Cluster Berdasarkan PCA")
            st.pyplot(fig)

            st.subheader("📊 Distribusi Klaster")
            st.bar_chart(df['Cluster'].value_counts().sort_index())

elif menu == "Evaluasi":
    st.title("📈 Evaluasi & Analisis Klaster")
    if st.session_state.clustered_data is None:
        st.warning("⚠️ Silakan lakukan klasterisasi terlebih dahulu.")
    else:
        df = st.session_state.clustered_data.copy()
        score = silhouette_score(df.drop(columns=['Cluster']), df['Cluster'])
        st.metric("Silhouette Score", f"{score:.4f}")

        if score >= 0.7:
            st.success("Hasil klasterisasi sangat baik, struktur klaster terbentuk dengan jelas.")
        elif score >= 0.5:
            st.info("Hasil klasterisasi cukup baik, namun masih ada sedikit tumpang tindih.")
        elif score >= 0.25:
            st.warning("Struktur klaster kurang optimal. Disarankan eksplorasi parameter lebih lanjut.")
        else:
            st.error("Struktur klaster lemah. Parameter mungkin belum sesuai dengan data.")

        st.subheader("📋 Ringkasan Statistik per Klaster")
        cluster_summary = df.groupby("Cluster").agg({
            'Jumlah Tenaga Kerja': 'mean',
            'Modal Awal': 'mean',
            'Luas Tanah': 'mean',
            'Skala Usaha': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'Risiko Usaha': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'Jenis Perusahaan': lambda x: x.mode()[0] if not x.mode().empty else np.nan
        }).round(2).reset_index()

        st.dataframe(cluster_summary)

        # Hitung rata-rata global
        global_mean = df[['Jumlah Tenaga Kerja', 'Modal Awal', 'Luas Tanah']].mean()

        st.subheader("🧠 Interpretasi Tiap Klaster")
        for _, row in cluster_summary.iterrows():
            cluster_id = int(row['Cluster'])
            tki = row['Jumlah Tenaga Kerja']
            modal = row['Modal Awal']
            tanah = row['Luas Tanah']
            skala = int(row['Skala Usaha'])
            risiko = int(row['Risiko Usaha'])
            jenis = int(row['Jenis Perusahaan'])

            skala_label = {0: 'Usaha Mikro', 1: 'Usaha Kecil', 2: 'Usaha Menengah'}.get(skala, 'Tidak Diketahui')
            risiko_label = {0: 'Rendah', 1: 'Menengah Rendah', 2: 'Menengah Tinggi', 3: 'Tinggi'}.get(risiko, 'Tidak Diketahui')
            jenis_label = {
                0: 'Perorangan', 1: 'Perseroan Terbatas (PT)', 2: 'CV', 3: 'PT Perorangan',
                4: 'Badan Hukum Lainnya', 5: 'Koperasi', 6: 'Firma (Fa)', 7: 'Perkumpulan',
                8: 'Yayasan', 9: 'Perumda', 10: 'BLU'
            }.get(jenis, 'Tidak Diketahui')

            tki_pos = 'banyak' if tki > global_mean['Jumlah Tenaga Kerja'] else 'sedikit'
            modal_pos = 'tinggi' if modal > global_mean['Modal Awal'] else 'rendah'
            tanah_pos = 'luas' if tanah > global_mean['Luas Tanah'] else 'sempit'

            st.markdown(f"""
            ### 🔹 Klaster {cluster_id}
            - Rata-rata **Jumlah Tenaga Kerja**: {tki:.2f} → tenaga kerja {tki_pos}
            - Rata-rata **Modal Awal**: {modal:.2f} → modal {modal_pos}
            - Rata-rata **Luas Tanah**: {tanah:.2f} → lahan {tanah_pos}
            - Dominan **Skala Usaha**: {skala_label}
            - Dominan **Risiko Usaha**: {risiko_label}
            - Dominan **Jenis Perusahaan**: {jenis_label}

            📌 **Kesimpulan**: Klaster ini merupakan kelompok dengan skala *{skala_label.lower()}*, modal {modal_pos}, dan tenaga kerja {tki_pos}. 
            Cocok untuk program {'perluasan pasar dan insentif investasi' if modal_pos == 'tinggi' else 'penguatan kapasitas usaha mikro dan kecil'}.
            """)

elif menu == "Download":
    st.title("⬇️ Unduh Hasil Klasterisasi")
    if st.session_state.clustered_data is None:
        st.warning("⚠️ Tidak ada hasil klasterisasi yang tersedia.")
    else:
        df = st.session_state.clustered_data.copy()
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Unduh sebagai CSV", csv, "hasil_klaster_ikm.csv", "text/csv")