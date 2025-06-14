import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
from cftree import CFTree
import joblib

st.set_page_config(page_title="IKMCluster Surabaya", layout="wide")

st.markdown("""
<style>
.stButton>button {
    background-color: #1976d2;
    color: white;
    padding: 0.6em 1.2em;
    border-radius: 8px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #0d47a1;
    color: white;
}
.boxed-title {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 6px;
    font-size: 18px;
    color: #0d3c61;
    font-weight: bold;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("IKMCluster Surabaya")
menu = st.sidebar.radio("Navigasi", ["About", "Upload Data", "Preprocessing", "Clustering", "Evaluasi", "Download"])

if "data" not in st.session_state:
    st.session_state.data = None
if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None

if menu == "About":
    st.title("📌 About Aplikasi")
    st.markdown("""
     **IKMCluster Surabaya** adalah aplikasi analitik interaktif untuk klasterisasi data Industri Kecil Menengah (IKM) Kota Surabaya menggunakan algoritma **BIRCH**. Aplikasi ini dirancang untuk membantu analisis kelompok usaha berdasarkan data karakteristik modal awal, luas tanah, jumlah tenaga kerja, skala usaha, risiko usaha, dan jenis perusahaan. Dengan pendekatan interaktif, pengguna dari berbagai latar belakang dapat memproses data, membentuk klaster, dan mengevaluasi hasilnya secara intuitif.

    Fitur utama:
    - Upload dan preprocessing data
    - Klasterisasi berbasis algoritma BIRCH
    - Visualisasi hasil klaster
    - Evaluasi kualitas klasterisasi
    - Interpretasi hasil klaster
    """)

elif menu == "Upload Data":
    st.title("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file dalam format CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success("✅ Data berhasil diunggah!")
        st.dataframe(df.head())

elif menu == "Preprocessing":
    st.title("🧹 Preprocessing Data")
    if st.session_state.data is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu.")
    else:
        df = st.session_state.data.copy()

        st.markdown("<div class='boxed-title'>📊 Statistik Deskriptif</div>", unsafe_allow_html=True)
        st.markdown("""
        Statistik deskriptif memberikan ringkasan umum mengenai data, seperti nilai rata-rata, standar deviasi, nilai minimum, dan maksimum dari variabel numerik. 
        Informasi ini penting untuk memahami skala dan sebaran data sebelum dilakukan analisis atau pemodelan lebih lanjut.
        """)
        st.dataframe(df.describe())

        st.markdown("<div class='boxed-title'>🩺 Cek dan Tangani Missing Values</div>", unsafe_allow_html=True)
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
        if missing.sum() > 0 and st.button("🧹 Hapus Baris Kosong"):
            df.dropna(inplace=True)
            st.success("✅ Missing values berhasil dihapus.")

        st.markdown("<div class='boxed-title'>🔠 Encoding Variabel Kategorik</div>", unsafe_allow_html=True)
        st.markdown("""
        Encoding adalah proses mengubah variabel kategorik menjadi bentuk numerik agar bisa diproses oleh algoritma machine learning. Karena sebagian besar algoritma hanya dapat bekerja dengan angka, encoding diperlukan untuk mengubah data seperti jenis usaha, risiko proyek, dan skala usaha menjadi nilai numerik.
        """)
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
            df['Risiko Proyek'] = df['Risiko Proyek'].map(risk_mapping)
            df['Jenis Perusahaan'] = df['Jenis Perusahaan'].map(company_mapping)
            st.success("✅ Encoding berhasil dilakukan.")
        except:
            st.error("Terjadi kesalahan dalam encoding.")

        st.markdown("<div class='boxed-title'>📏 Standarisasi Variabel Numerik</div>", unsafe_allow_html=True)
        st.markdown("""
        Standarisasi adalah proses untuk menyamakan skala semua variabel numerik agar memiliki pengaruh yang seimbang dalam proses pengelompokan (klastering). Tanpa standarisasi, variabel dengan nilai yang lebih besar bisa mendominasi hasil. Standarisasi dilakukan menggunakan metode Standard Scaler, yaitu dengan mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.
        """)
        scaler = StandardScaler()
        num_cols = ['Tenaga Kerja', 'Modal Awal', 'Luas Tanah']
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.session_state.data = df
        st.dataframe(df.head())

elif menu == "Clustering":
    st.title("🌿 Klasterisasi IKM (BIRCH)")
    if st.session_state.data is None:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu.")
    else:
        X = st.session_state.data.copy()
        st.markdown("<div class='boxed-title'>🧠 Pilih Metode</div>", unsafe_allow_html=True)
        mode = st.radio("Gunakan model yang mana?", ["Gunakan model .pkl (siap pakai)", "Bangun model baru (atur parameter)"])

        if mode == "Gunakan model .pkl (siap pakai)":
            model = joblib.load("birch_model.pkl")
            # Ambil hanya kolom numerik yang dipakai untuk clustering
            fitur_model = ['Tenaga Kerja', 'Modal Awal', 'Luas Tanah', 'Skala Usaha', 'Risiko Proyek', 'Jenis Perusahaan']
            X_fit = X[fitur_model].astype(float).to_numpy()
            # Prediksi
            labels = model.predict(X_fit)
            # Tambahkan label ke DataFrame asli
            X['Cluster'] = labels
            st.session_state.clustered_data = X
            st.success("✅ Klasterisasi selesai menggunakan model yang telah disimpan.")

        else:
            st.markdown("""
            <div class='boxed-title'>⚙️ Parameter BIRCH</div>
            <p style='margin-bottom:10px'>
            <b>Threshold</b>: batas maksimum jarak antar data agar dianggap mirip dan dimasukkan ke dalam klaster yang sama. Semakin besar nilainya, semakin longgar aturan pengelompokannya, sehingga jumlah klaster yang terbentuk cenderung lebih sedikit.
            </p>
            <p><b>Branching Factor</b>: jumlah maksimum cabang yang boleh dimiliki oleh setiap simpul dalam struktur pohon. Semakin besar nilainya, struktur pohonnya akan semakin kompleks dan bisa memuat lebih banyak data dalam satu simpul.</p>
            """, unsafe_allow_html=True)
            threshold = st.slider("Threshold", 0.1, 2.0, 0.5, step=0.1)
            branching_factor = st.slider("Branching Factor", 10, 100, 50, step=10)
            if st.button("🚀 Jalankan Clustering"):
                birch = Birch(threshold=threshold, branching_factor=branching_factor)
                cluster_labels = birch.fit_predict(X)
                X['Cluster'] = cluster_labels
                st.session_state.clustered_data = X

                st.success(f"✅ Clustering selesai! Jumlah klaster terbentuk: {X['Cluster'].nunique()}")
                st.dataframe(X.head())

        st.markdown("<div class='boxed-title'>📊 Visualisasi PCA 3D</div>", unsafe_allow_html=True)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(X.drop(columns=['Cluster']))
        df_pca = pd.DataFrame(reduced, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = X['Cluster']
        fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.8,
                            title='Visualisasi Klaster dalam Ruang 3D', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Distribusi Klaster")
        st.bar_chart(X['Cluster'].value_counts().sort_index())

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
            'Tenaga Kerja': 'mean',
            'Modal Awal': 'mean',
            'Luas Tanah': 'mean',
            'Skala Usaha': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'Risiko Proyek': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'Jenis Perusahaan': lambda x: x.mode()[0] if not x.mode().empty else np.nan
        }).round(2).reset_index()

        st.dataframe(cluster_summary)

        global_mean = df[['Tenaga Kerja', 'Modal Awal', 'Luas Tanah']].mean()

        st.subheader("🧠 Interpretasi Tiap Klaster")
        for _, row in cluster_summary.iterrows():
            cluster_id = int(row['Cluster'])
            tenaga_kerja = row['Tenaga Kerja']
            modal = row['Modal Awal']
            tanah = row['Luas Tanah']
            skala = int(row['Skala Usaha'])
            risiko = int(row['Risiko Proyek'])
            jenis = int(row['Jenis Perusahaan'])

            skala_label = {0: 'Usaha Mikro', 1: 'Usaha Kecil', 2: 'Usaha Menengah'}.get(skala, 'Tidak Diketahui')
            risiko_label = {0: 'Rendah', 1: 'Menengah Rendah', 2: 'Menengah Tinggi', 3: 'Tinggi'}.get(risiko, 'Tidak Diketahui')
            jenis_label = {
                0: 'Perorangan', 1: 'Perseroan Terbatas (PT)', 2: 'CV', 3: 'PT Perorangan',
                4: 'Badan Hukum Lainnya', 5: 'Koperasi', 6: 'Firma (Fa)', 7: 'Perkumpulan',
                8: 'Yayasan', 9: 'Perumda', 10: 'BLU'
            }.get(jenis, 'Tidak Diketahui')

            tenagakerja_pos = 'banyak' if tenaga_kerja > global_mean['Tenaga Kerja'] else 'sedikit'
            modal_pos = 'tinggi' if modal > global_mean['Modal Awal'] else 'rendah'
            tanah_pos = 'luas' if tanah > global_mean['Luas Tanah'] else 'sempit'

            st.markdown(f"""
            ### 🔹 Klaster {cluster_id}
            - Rata-rata **Jumlah Tenaga Kerja (TKI)**: {tenaga_kerja:.2f} → tenaga kerja {tenagakerja_pos}
            - Rata-rata **Modal Awal**: {modal:.2f} → modal {modal_pos}
            - Rata-rata **Luas Tanah**: {tanah:.2f} → lahan {tanah_pos}
            - Dominan **Skala Usaha**: {skala_label}
            - Dominan **Risiko Proyek**: {risiko_label}
            - Dominan **Jenis Perusahaan**: {jenis_label}

            📌 **Kesimpulan**: Klaster ini merupakan kelompok dengan skala *{skala_label.lower()}*, modal {modal_pos}, dan tenaga kerja {tenagakerja_pos}. 
            Cocok untuk program {'perluasan pasar dan insentif investasi' if modal_pos == 'tinggi' else 'penguatan kapasitas usaha mikro dan kecil'}.
            """)

elif menu == "Download":
    st.title("⬇️ Unduh Hasil Klasterisasi")
    if st.session_state.clustered_data is not None:
        df = st.session_state.clustered_data.copy()
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Unduh sebagai CSV", csv, "hasil_klaster.csv", "text/csv")
    else:
        st.warning("⚠️ Tidak ada hasil klasterisasi yang tersedia.")
