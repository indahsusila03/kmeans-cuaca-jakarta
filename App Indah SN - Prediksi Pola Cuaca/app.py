"""
Aplikasi K-Means Clustering Cuaca Harian
==========================================
Implementasi web (Streamlit) dari notebook analisis klaster cuaca BMKG.
Alur: Upload/Gabung Data -> Bersihkan & Normalisasi -> Tentukan K Optimal
      -> Clustering -> Visualisasi -> Prediksi Data Baru
"""

import io
import re
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="K-Means Klaster Cuaca",
    page_icon="🌦️",
    layout="wide",
)

KOLOM_CUACA_DEFAULT = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'FF_AVG']
KOLOM_CLUSTERING_DEFAULT = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'FF_AVG']
LABEL_PARAM = {
    'TN': 'Suhu Minimum (°C)', 'TX': 'Suhu Maksimum (°C)',
    'TAVG': 'Suhu Rata-rata (°C)', 'RH_AVG': 'Kelembapan Relatif (%)',
    'RR': 'Curah Hujan (mm)', 'SS': 'Lama Penyinaran Matahari (jam)',
    'FF_X': 'Kec. Angin Maksimum (m/s)', 'FF_AVG': 'Kec. Angin Rata-rata (m/s)',
}
WARNA_PALET = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#E53935',
               '#00BCD4', '#8BC34A', '#FFC107', '#795548', '#607D8B']

# =====================================================================
# STATE
# =====================================================================
def init_state():
    defaults = {
        'data_gabungan': None,
        'kolom_cuaca': None,
        'X_scaled': None,
        'scaler': None,
        'kolom_clustering': KOLOM_CLUSTERING_DEFAULT,
        'wcss': None,
        'sil_scores': None,
        'k_range': None,
        'kmeans_model': None,
        'k_terpilih': None,
        'nama_klaster': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# =====================================================================
# FUNGSI INTI (diadaptasi langsung dari notebook)
# =====================================================================
def baca_file_bmkg(file_obj, filename: str) -> pd.DataFrame:
    """Membaca satu file Excel BMKG, mencari baris header otomatis."""
    raw = pd.read_excel(file_obj, header=None)

    baris_header = None
    for i, baris in raw.iterrows():
        nilai = str(baris.values)
        if 'TANGGAL' in nilai or 'DATE' in nilai:
            baris_header = i
            break

    if baris_header is None:
        return pd.DataFrame()

    file_obj.seek(0)
    df = pd.read_excel(file_obj, header=baris_header)
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')

    if 'DATE' in df.columns and 'TANGGAL' not in df.columns:
        df = df.rename(columns={'DATE': 'TANGGAL'})

    df['TANGGAL'] = df['TANGGAL'].astype(str).str.strip()

    def is_tanggal_valid(nilai):
        return bool(re.match(r'\d{2}-\d{2}-\d{4}', str(nilai)))

    df = df[df['TANGGAL'].apply(is_tanggal_valid)]

    bulan = filename.replace('.xlsx', '').replace('.xls', '')
    bulan = re.sub(r'_?\d{4}$', '', bulan).strip('_')
    df['BULAN'] = bulan if bulan else filename

    return df


def bersihkan_dan_imputasi(df: pd.DataFrame):
    """Konversi ke numerik, buang kode error BMKG, isi NaN dengan median per bulan."""
    kolom_hapus = ['DDD_X', 'DDD_CAR']
    df = df.drop(columns=[c for c in kolom_hapus if c in df.columns], errors='ignore')

    kolom_cuaca = [c for c in KOLOM_CUACA_DEFAULT if c in df.columns]

    df[kolom_cuaca] = df[kolom_cuaca].apply(pd.to_numeric, errors='coerce')
    df[kolom_cuaca] = df[kolom_cuaca].replace([8888, 9999], np.nan)

    for kolom in kolom_cuaca:
        df[kolom] = df.groupby('BULAN')[kolom].transform(lambda x: x.fillna(x.median()))
        # jika seluruh bulan NaN, isi dengan median global
        df[kolom] = df[kolom].fillna(df[kolom].median())

    return df, kolom_cuaca


def buat_data_contoh(n_hari: int = 365, seed: int = 42) -> pd.DataFrame:
    """Membuat data cuaca sintetis (untuk demo cepat tanpa perlu upload file)."""
    rng = np.random.default_rng(seed)
    tanggal = pd.date_range('2025-01-01', periods=n_hari, freq='D')

    # bobot musim kemarau vs hujan berdasarkan bulan (kasar, gaya Jakarta)
    bulan_hujan = {1, 2, 3, 11, 12}
    rows = []
    for tgl in tanggal:
        musim_hujan = tgl.month in bulan_hujan
        acak = rng.random()
        if musim_hujan and acak < 0.35:
            profil = 'hujan'
        elif not musim_hujan and acak < 0.55:
            profil = 'panas'
        else:
            profil = 'sejuk'

        if profil == 'panas':
            tavg = rng.normal(29.5, 0.8); rh = rng.normal(62, 5); rr = max(0, rng.normal(1, 2))
        elif profil == 'hujan':
            tavg = rng.normal(26.0, 0.7); rh = rng.normal(88, 4); rr = max(0, rng.normal(35, 20))
        else:
            tavg = rng.normal(27.2, 0.6); rh = rng.normal(78, 5); rr = max(0, rng.normal(6, 5))

        tn = tavg - rng.uniform(2, 4)
        tx = tavg + rng.uniform(2, 5)
        ff_avg = max(0.2, rng.normal(2.3, 0.7))
        ff_x = ff_avg + rng.uniform(1, 3)
        ss = max(0, rng.normal(6 if profil == 'panas' else 3, 2))

        rows.append({
            'TANGGAL': tgl.strftime('%d-%m-%Y'),
            'BULAN': tgl.strftime('%B %Y').lower(),
            'TN': round(tn, 1), 'TX': round(tx, 1), 'TAVG': round(tavg, 1),
            'RH_AVG': round(np.clip(rh, 40, 100), 1), 'RR': round(rr, 1),
            'SS': round(ss, 1), 'FF_X': round(ff_x, 1), 'FF_AVG': round(ff_avg, 1),
        })

    return pd.DataFrame(rows)


def saran_nama_klaster(profil: pd.DataFrame) -> dict:
    """Heuristik pemberian nama awal klaster berdasarkan karakteristik curah hujan & suhu."""
    nama = {}
    sisa = list(profil.index)

    if 'RR' in profil.columns and len(sisa) > 0:
        k_hujan = profil.loc[sisa, 'RR'].idxmax()
        nama[k_hujan] = 'Cuaca Hujan Lebat'
        sisa.remove(k_hujan)

    if 'TAVG' in profil.columns and len(sisa) > 0:
        k_panas = profil.loc[sisa, 'TAVG'].idxmax()
        nama[k_panas] = 'Cuaca Panas Kering'
        sisa.remove(k_panas)

    for i, k in enumerate(sisa, start=1):
        nama[k] = f'Klaster Cuaca {i}' if len(sisa) > 1 else 'Cuaca Sejuk Berawan'

    return nama


# =====================================================================
# SIDEBAR — status pipeline
# =====================================================================
with st.sidebar:
    st.title("🌦️ K-Means Klaster Cuaca")
    st.caption("Implementasi analisis klaster cuaca harian BMKG")
    st.divider()
    st.subheader("Status Pipeline")
    st.write("1. Data" + (" ✅" if st.session_state.data_gabungan is not None else " ⬜"))
    st.write("2. Normalisasi" + (" ✅" if st.session_state.X_scaled is not None else " ⬜"))
    st.write("3. K Optimal" + (" ✅" if st.session_state.sil_scores is not None else " ⬜"))
    st.write("4. Model Klaster" + (" ✅" if st.session_state.kmeans_model is not None else " ⬜"))

    if st.session_state.data_gabungan is not None:
        st.divider()
        st.metric("Jumlah hari (data)", len(st.session_state.data_gabungan))

    st.divider()
    if st.session_state.kmeans_model is not None:
        buf = io.BytesIO()
        pickle.dump({
            'scaler': st.session_state.scaler,
            'model': st.session_state.kmeans_model,
            'kolom_clustering': st.session_state.kolom_clustering,
            'nama_klaster': st.session_state.nama_klaster,
        }, buf)
        st.download_button(
            "⬇️ Unduh Model (.pkl)", data=buf.getvalue(),
            file_name="model_klaster_cuaca.pkl", mime="application/octet-stream",
            use_container_width=True,
        )
    st.caption("Muat model tersimpan di tab **4. Prediksi**")


# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Upload & Data", "2️⃣ K Optimal", "3️⃣ Clustering & Visualisasi", "4️⃣ Prediksi Data Baru",
])

# ---------------------------------------------------------------------
# TAB 1 — UPLOAD & DATA
# ---------------------------------------------------------------------
with tab1:
    st.header("Upload dan Persiapan Data")
    st.write(
        "Unggah file Excel data harian BMKG (satu file per bulan). Aplikasi otomatis "
        "mendeteksi baris header, menggabungkan seluruh file, membersihkan kode error "
        "(8888/9999), dan mengisi nilai kosong dengan median per bulan."
    )

    colA, colB = st.columns([2, 1])
    with colA:
        uploaded_files = st.file_uploader(
            "Pilih file Excel BMKG (.xlsx)", type=['xlsx', 'xls'], accept_multiple_files=True,
        )
    with colB:
        st.write("")
        st.write("")
        pakai_contoh = st.button("🎲 Gunakan Data Contoh (Demo)", use_container_width=True)

    if pakai_contoh:
        df_contoh = buat_data_contoh()
        df_bersih, kolom_cuaca = bersihkan_dan_imputasi(df_contoh)
        st.session_state.data_gabungan = df_bersih
        st.session_state.kolom_cuaca = kolom_cuaca
        st.session_state.X_scaled = None
        st.session_state.kmeans_model = None
        st.success(f"Data contoh dibuat: {len(df_bersih)} hari (sintetis, hanya untuk demo).")

    if uploaded_files:
        if st.button("🔄 Proses & Gabungkan File", type="primary"):
            semua_data, log = [], []
            for f in uploaded_files:
                try:
                    df = baca_file_bmkg(f, f.name)
                    if len(df) > 0:
                        semua_data.append(df)
                        log.append(f"✅ {f.name} → {len(df)} hari")
                    else:
                        log.append(f"⚠️ {f.name} → 0 hari (header tidak ditemukan)")
                except Exception as e:
                    log.append(f"❌ {f.name} → ERROR: {e}")

            with st.expander("Log proses file", expanded=True):
                for line in log:
                    st.write(line)

            if semua_data:
                data_gabungan = pd.concat(semua_data, ignore_index=True)
                data_gabungan, kolom_cuaca = bersihkan_dan_imputasi(data_gabungan)
                st.session_state.data_gabungan = data_gabungan
                st.session_state.kolom_cuaca = kolom_cuaca
                st.session_state.X_scaled = None
                st.session_state.kmeans_model = None
                st.success(f"🎉 Total data tergabung: {len(data_gabungan)} hari")
            else:
                st.error("Tidak ada file yang berhasil diproses. Periksa format file Anda.")

    st.divider()

    if st.session_state.data_gabungan is not None:
        df = st.session_state.data_gabungan
        kolom_cuaca = st.session_state.kolom_cuaca

        st.subheader("Pratinjau Data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total hari", len(df))
        c2.metric("Jumlah bulan", df['BULAN'].nunique())
        c3.metric("Parameter cuaca", len(kolom_cuaca))
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Ringkasan Statistik")
        st.dataframe(df[kolom_cuaca].describe().round(2), use_container_width=True)

        st.subheader("Normalisasi Data (StandardScaler)")
        if st.button("⚙️ Normalisasi Data"):
            X = df[kolom_cuaca].copy()
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=kolom_cuaca)
            st.session_state.scaler = scaler
            st.session_state.X_scaled = X_scaled
            st.success("Data berhasil dinormalisasi. Lanjut ke tab '2️⃣ K Optimal'.")

        if st.session_state.X_scaled is not None:
            st.dataframe(
                st.session_state.X_scaled.describe().round(2).loc[['mean', 'std', 'min', 'max']],
                use_container_width=True,
            )
    else:
        st.info("Belum ada data. Unggah file BMKG atau gunakan data contoh di atas.")


# ---------------------------------------------------------------------
# TAB 2 — K OPTIMAL
# ---------------------------------------------------------------------
with tab2:
    st.header("Penentuan Jumlah Klaster (K) Optimal")

    if st.session_state.X_scaled is None:
        st.warning("Normalisasi data terlebih dahulu di tab '1️⃣ Upload & Data'.")
    else:
        kolom_tersedia = st.session_state.kolom_cuaca
        default_pilihan = [c for c in KOLOM_CLUSTERING_DEFAULT if c in kolom_tersedia] or kolom_tersedia

        kolom_clustering = st.multiselect(
            "Parameter yang digunakan untuk clustering",
            options=kolom_tersedia, default=default_pilihan,
            format_func=lambda c: LABEL_PARAM.get(c, c),
        )
        st.session_state.kolom_clustering = kolom_clustering

        k_min, k_max = st.slider("Rentang K yang diuji", 2, 10, (2, 10))

        if kolom_clustering and st.button("📈 Hitung WCSS & Silhouette Score", type="primary"):
            X_cluster = st.session_state.X_scaled[kolom_clustering]
            K_range = range(k_min, k_max + 1)
            wcss, sil_scores = [], []

            progress = st.progress(0.0)
            for i, k in enumerate(K_range):
                km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
                km.fit(X_cluster)
                wcss.append(km.inertia_)
                sil_scores.append(silhouette_score(X_cluster, km.labels_))
                progress.progress((i + 1) / len(list(K_range)))

            st.session_state.wcss = wcss
            st.session_state.sil_scores = sil_scores
            st.session_state.k_range = list(K_range)

        if st.session_state.sil_scores is not None:
            K_range = st.session_state.k_range
            wcss = st.session_state.wcss
            sil_scores = st.session_state.sil_scores
            k_terbaik = K_range[int(np.argmax(sil_scores))]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
            ax1.plot(K_range, wcss, 'bo-', linewidth=2, markersize=7)
            ax1.axvline(x=k_terbaik, color='red', linestyle='--', linewidth=1.5,
                        label=f'K disarankan = {k_terbaik}')
            ax1.set_title('Metode Elbow', fontweight='bold')
            ax1.set_xlabel('Jumlah Klaster (K)'); ax1.set_ylabel('WCSS')
            ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(list(K_range))

            ax2.plot(K_range, sil_scores, 'rs-', linewidth=2, markersize=7)
            ax2.axvline(x=k_terbaik, color='blue', linestyle='--', linewidth=1.5,
                        label=f'K disarankan = {k_terbaik}')
            ax2.set_title('Silhouette Score', fontweight='bold')
            ax2.set_xlabel('Jumlah Klaster (K)'); ax2.set_ylabel('Silhouette Score')
            ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(list(K_range))

            plt.tight_layout()
            st.pyplot(fig)

            st.info(
                f"💡 Berdasarkan Silhouette Score tertinggi, **K = {k_terbaik}** adalah "
                f"jumlah klaster yang disarankan (skor = {max(sil_scores):.4f})."
            )

            k_pilih = st.number_input(
                "Pilih K untuk digunakan pada tahap clustering", min_value=min(K_range),
                max_value=max(K_range), value=int(k_terbaik), step=1,
            )
            st.session_state.k_terpilih = int(k_pilih)
            st.success(f"K = {k_pilih} siap digunakan. Lanjut ke tab '3️⃣ Clustering & Visualisasi'.")


# ---------------------------------------------------------------------
# TAB 3 — CLUSTERING & VISUALISASI
# ---------------------------------------------------------------------
with tab3:
    st.header("Clustering & Visualisasi Hasil")

    if st.session_state.k_terpilih is None:
        st.warning("Tentukan K optimal terlebih dahulu di tab '2️⃣ K Optimal'.")
    else:
        k = st.session_state.k_terpilih
        kolom_clustering = st.session_state.kolom_clustering
        X_cluster = st.session_state.X_scaled[kolom_clustering]
        df = st.session_state.data_gabungan

        if st.button(f"🚀 Jalankan K-Means (K={k})", type="primary"):
            kmeans_final = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
            kmeans_final.fit(X_cluster)
            df['KLASTER'] = kmeans_final.labels_
            st.session_state.kmeans_model = kmeans_final
            st.session_state.data_gabungan = df

            profil = df.groupby('KLASTER')[kolom_clustering].mean().round(2)
            st.session_state.nama_klaster = saran_nama_klaster(profil)

        if st.session_state.kmeans_model is not None:
            kmeans_final = st.session_state.kmeans_model
            df = st.session_state.data_gabungan
            profil = df.groupby('KLASTER')[kolom_clustering].mean().round(2)

            st.subheader("Nama Klaster")
            st.caption("Nama disarankan otomatis berdasarkan karakteristik — silakan sesuaikan.")
            cols = st.columns(k)
            for i, kk in enumerate(sorted(df['KLASTER'].unique())):
                default_nama = st.session_state.nama_klaster.get(kk, f'Klaster {kk}')
                with cols[i % k]:
                    nama_baru = st.text_input(f"Klaster {kk}", value=default_nama, key=f"nama_{kk}")
                    st.session_state.nama_klaster[kk] = nama_baru

            df['NAMA_KLASTER'] = df['KLASTER'].map(st.session_state.nama_klaster)
            st.session_state.data_gabungan = df
            warna = {kk: WARNA_PALET[i % len(WARNA_PALET)] for i, kk in enumerate(sorted(df['KLASTER'].unique()))}

            st.subheader("Karakteristik Tiap Klaster (nilai asli)")
            st.dataframe(profil, use_container_width=True)

            jumlah_klaster = df['KLASTER'].value_counts().sort_index()
            cols_metric = st.columns(k)
            for i, kk in enumerate(sorted(df['KLASTER'].unique())):
                jumlah = jumlah_klaster[kk]
                persen = jumlah / len(df) * 100
                cols_metric[i].metric(st.session_state.nama_klaster[kk], f"{jumlah} hari", f"{persen:.1f}%")

            st.divider()
            st.subheader("Visualisasi")

            viz1, viz2 = st.columns(2)
            with viz1:
                fig, ax = plt.subplots(figsize=(6, 5))
                labels_pie = [f"{st.session_state.nama_klaster[kk]}" for kk in jumlah_klaster.index]
                ax.pie(jumlah_klaster, labels=labels_pie,
                       colors=[warna[kk] for kk in jumlah_klaster.index],
                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
                ax.set_title('Proporsi Hari per Klaster', fontweight='bold')
                st.pyplot(fig)

            with viz2:
                fig, ax = plt.subplots(figsize=(6.5, 5))
                param_bar = [c for c in ['TAVG', 'RH_AVG', 'FF_AVG'] if c in kolom_clustering][:3] or kolom_clustering[:3]
                profil_plot = df.groupby('NAMA_KLASTER')[param_bar].mean().round(2)
                profil_plot.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.5)
                ax.set_title('Rata-rata Parameter per Klaster', fontweight='bold')
                ax.set_xlabel(''); ax.tick_params(axis='x', rotation=15)
                ax.legend([LABEL_PARAM.get(p, p) for p in param_bar], fontsize=8)
                ax.grid(alpha=0.3, axis='y')
                st.pyplot(fig)

            viz3, viz4 = st.columns(2)
            with viz3:
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                korelasi = df[kolom_clustering].corr().round(2)
                sns.heatmap(korelasi, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                            linewidths=0.5, ax=ax, annot_kws={'size': 9})
                ax.set_title('Heatmap Korelasi Parameter', fontweight='bold')
                st.pyplot(fig)

            with viz4:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_cluster)
                var1, var2 = pca.explained_variance_ratio_[:2] * 100

                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                for kk in sorted(df['KLASTER'].unique()):
                    mask = df['KLASTER'] == kk
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=warna[kk],
                               label=st.session_state.nama_klaster[kk], alpha=0.6, s=45,
                               edgecolors='white', linewidth=0.4)
                centroids_pca = pca.transform(kmeans_final.cluster_centers_)
                for i, kk in enumerate(sorted(df['KLASTER'].unique())):
                    ax.scatter(centroids_pca[i, 0], centroids_pca[i, 1], c=warna[kk], marker='*',
                               s=260, edgecolors='black', linewidth=1, zorder=5)
                ax.set_xlabel(f'PC1 ({var1:.1f}%)'); ax.set_ylabel(f'PC2 ({var2:.1f}%)')
                ax.set_title('Scatter Plot Klaster (PCA)', fontweight='bold')
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
                st.pyplot(fig)

            if 'BULAN' in df.columns:
                st.subheader("Distribusi Klaster per Bulan")
                bulan_klaster = df.groupby(['BULAN', 'NAMA_KLASTER']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(12, 5))
                bulan_klaster.plot(kind='bar', stacked=True, ax=ax, edgecolor='white', linewidth=0.5)
                ax.set_title('Distribusi Klaster Cuaca per Bulan', fontweight='bold')
                ax.set_xlabel('Bulan'); ax.set_ylabel('Jumlah Hari')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Klaster', bbox_to_anchor=(1.01, 1), loc='upper left')
                ax.grid(alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)

            st.divider()
            st.subheader("Unduh Hasil")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Unduh Data Berklaster (.csv)", data=csv,
                                file_name="data_cuaca_berklaster.csv", mime="text/csv")


# ---------------------------------------------------------------------
# TAB 4 — PREDIKSI DATA BARU
# ---------------------------------------------------------------------
with tab4:
    st.header("Prediksi Klaster untuk Data Cuaca Baru")
    st.write(
        "Gunakan model yang baru dilatih (tab 3), atau muat model `.pkl` yang pernah "
        "diunduh sebelumnya, lalu masukkan parameter cuaca satu hari untuk melihat "
        "termasuk klaster apa harinya."
    )

    model_pkl = st.file_uploader("Atau muat model tersimpan (.pkl)", type=['pkl'])
    if model_pkl is not None:
        loaded = pickle.load(model_pkl)
        st.session_state.scaler = loaded['scaler']
        st.session_state.kmeans_model = loaded['model']
        st.session_state.kolom_clustering = loaded['kolom_clustering']
        st.session_state.nama_klaster = loaded['nama_klaster']
        st.success("Model berhasil dimuat.")

    if st.session_state.kmeans_model is None or st.session_state.scaler is None:
        st.warning("Belum ada model. Latih model di tab 3, atau muat file .pkl di atas.")
    else:
        kolom_clustering = st.session_state.kolom_clustering
        st.subheader("Masukkan Parameter Cuaca")

        input_vals = {}
        cols = st.columns(min(3, len(kolom_clustering)))
        default_ranges = {
            'TN': (24.0, 0.1), 'TX': (32.0, 0.1), 'TAVG': (28.0, 0.1),
            'RH_AVG': (75.0, 0.5), 'RR': (5.0, 0.5), 'FF_AVG': (2.5, 0.1), 'SS': (5.0, 0.1), 'FF_X': (5.0, 0.1),
        }
        for i, kolom in enumerate(kolom_clustering):
            default_val, step = default_ranges.get(kolom, (0.0, 0.1))
            with cols[i % len(cols)]:
                input_vals[kolom] = st.number_input(
                    LABEL_PARAM.get(kolom, kolom), value=default_val, step=step, key=f"pred_{kolom}",
                )

        if st.button("🔮 Prediksi Klaster", type="primary"):
            # Scaler dilatih pada seluruh kolom_cuaca; scaling manual di sini hanya
            # mengambil mean/scale untuk kolom_clustering sesuai indeksnya masing-masing.
            scaler = st.session_state.scaler
            kolom_cuaca_scaler = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else kolom_clustering
            idx = [kolom_cuaca_scaler.index(c) for c in kolom_clustering if c in kolom_cuaca_scaler]
            nilai_input = np.array([[input_vals[c] for c in kolom_clustering]])

            if len(idx) == len(kolom_clustering):
                mean_ = scaler.mean_[idx]
                scale_ = scaler.scale_[idx]
                nilai_scaled = (nilai_input - mean_) / scale_
            else:
                nilai_scaled = nilai_input  # fallback jika scaler tidak menyimpan nama fitur

            nilai_scaled = pd.DataFrame(nilai_scaled, columns=kolom_clustering)

            model = st.session_state.kmeans_model
            klaster_pred = model.predict(nilai_scaled)[0]
            jarak = model.transform(nilai_scaled)[0]
            nama = st.session_state.nama_klaster.get(klaster_pred, f"Klaster {klaster_pred}")

            st.success(f"### Hasil: **Klaster {klaster_pred} — {nama}**")

            jarak_df = pd.DataFrame({
                'Klaster': [st.session_state.nama_klaster.get(i, f"Klaster {i}") for i in range(len(jarak))],
                'Jarak ke Centroid': jarak.round(3),
            }).sort_values('Jarak ke Centroid')
            st.write("Jarak ke tiap centroid (semakin kecil = semakin mirip):")
            st.dataframe(jarak_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Aplikasi ini adalah implementasi dari notebook analisis K-Means clustering data cuaca harian. "
    f"Dibuat {datetime.now().strftime('%Y')}."
)
