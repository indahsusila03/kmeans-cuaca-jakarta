# Aplikasi K-Means Klaster Cuaca Harian

Implementasi web (Streamlit) dari notebook analisis K-Means clustering data cuaca
harian BMKG. Aplikasi ini membungkus seluruh alur kerja notebook — mulai dari
menggabungkan file Excel BMKG, membersihkan data, normalisasi, penentuan K optimal,
clustering, visualisasi, sampai prediksi klaster untuk data baru — ke dalam satu
aplikasi interaktif yang bisa dipakai tanpa menulis kode.

## Fitur

1. **Upload & Data** — unggah beberapa file Excel BMKG sekaligus (deteksi header
   otomatis), atau gunakan tombol "Data Contoh" untuk mencoba aplikasi tanpa file.
2. **K Optimal** — hitung dan lihat grafik Elbow serta Silhouette Score untuk
   memilih jumlah klaster (K) terbaik.
3. **Clustering & Visualisasi** — jalankan K-Means, beri nama tiap klaster, lihat
   pie chart, bar chart, heatmap korelasi, scatter plot PCA, dan distribusi
   klaster per bulan. Hasil bisa diunduh sebagai CSV.
4. **Prediksi Data Baru** — masukkan parameter cuaca satu hari (suhu, kelembapan,
   curah hujan, dll) dan langsung lihat termasuk klaster cuaca apa hari tersebut.
   Model terlatih juga bisa diunduh (`.pkl`) dan dimuat kembali di sesi lain tanpa
   perlu melatih ulang.

## Cara Menjalankan

1. Pastikan Python 3.9+ sudah terpasang.
2. Buka terminal di folder ini, lalu buat virtual environment (opsional tapi disarankan):

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Pasang seluruh dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

5. Browser akan otomatis terbuka di `http://localhost:8501`. Jika tidak, buka
   alamat tersebut secara manual.

## Format File Input

File Excel yang diunggah sebaiknya mengikuti format ekspor data harian BMKG,
dengan kolom seperti: `TANGGAL`, `TN`, `TX`, `TAVG`, `RH_AVG`, `RR`, `SS`,
`FF_X`, `FF_AVG`. Nama bulan diambil otomatis dari nama file (mis.
`januari_2025.xlsx` → bulan "januari").

## Catatan

- Kode 8888/9999 (kode "data tidak tersedia" BMKG) otomatis diubah menjadi nilai
  kosong, lalu diisi dengan median per bulan.
- Model yang diunduh (`.pkl`) menyimpan scaler, model K-Means, kolom yang
  digunakan untuk clustering, dan nama klaster — sehingga bisa dipakai ulang
  di tab "Prediksi Data Baru" tanpa perlu upload data mentah lagi.
