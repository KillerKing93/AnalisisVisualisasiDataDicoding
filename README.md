# Dashboard Kualitas Udara Tionghoa

## Deskripsi

Projek ini merupakan dashboard interaktif berbasis [Streamlit](https://streamlit.io/) untuk menganalisis data kualitas udara dari beberapa stasiun di Tionghoa. Fitur-fitur yang tersedia meliputi:

- **Peta Geospasial:** Menampilkan peta interaktif konsentrasi PM2.5 menggunakan [Folium](https://python-visualization.github.io/folium/).
- **Visualisasi Data:** Menyediakan tab visualisasi untuk:
  - **Statistik Stasiun:** Menampilkan statistik deskriptif, pie chart, heatmap korelasi, dan facet plot tren polutan.
  - **Tren & Hubungan:** Menampilkan tren konsentrasi polutan (PM2.5, PM10) serta hubungan antara polutan dengan variabel meteorologi.
  - **Perbandingan Antar Stasiun:** Memungkinkan perbandingan antar stasiun untuk berbagai variabel (PM, meteorologi, dan polutan lainnya).
- **Filter Interaktif:** Pengguna dapat memfilter data berdasarkan stasiun dan rentang tanggal.

Selain dashboard interaktif, terdapat pula file **notebook.ipynb** yang menyediakan analisis data lebih mendalam menggunakan Jupyter Notebook.

## Instalasi

Pastikan Anda sudah menginstall Python (minimal versi 3.7) di sistem Anda.

### 1. Membuat Virtual Environment

Di direktori root proyek, buat virtual environment dengan perintah berikut:

```bash
python3 -m venv venv
- atau
python -m venv venv
```

### 2. Mengaktifkan Virtual Environment

Aktifkan virtual environment yang telah dibuat:

- **Untuk Linux/MacOS:**
  ```bash
  source venv/bin/activate
  ```
- **Untuk Windows:**
  ```bash
  venv\Scripts\activate
  ```

> **Catatan:** Pastikan virtual environment aktif di direktori root proyek agar semua path relatif (seperti pemanggilan dataset dan gambar) berfungsi dengan baik.

### 3. Instalasi Dependencies

Setelah virtual environment aktif, instal semua dependencies yang diperlukan dengan menjalankan perintah:

```bash
pip install -r requirements.txt
```

Pastikan file `requirements.txt` sudah terisi dengan paket-paket berikut (atau paket lain sesuai kebutuhan proyek):

- streamlit
- pandas
- seaborn
- matplotlib
- numpy
- scipy
- folium
- streamlit_folium
- jupyter

## Cara Menjalankan Projek

### Menjalankan Dashboard Streamlit

Untuk menjalankan dashboard interaktif, jalankan perintah berikut di direktori root:

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka secara otomatis di browser Anda.

### Menjalankan Jupyter Notebook

Untuk melihat analisis data secara lebih mendalam, jalankan Jupyter Notebook dengan perintah:

```bash
jupyter notebook notebook.ipynb
```

Kemudian, buka file `notebook.ipynb` melalui antarmuka web Jupyter Notebook yang terbuka.

## Struktur Folder Proyek

```
├── dashboard/
│   ├── dashboard.py           # Kode dashboard Streamlit
│   └── PSRA_Data_SemuaStasiun.csv  # Gabungan dataset mentah yang telah dibersihkan dan digunakan pada dashboard
├── data/                     # Dataset mentah
│   ├── PRSA_Data_Aotizhongxin.csv
│   ├── PRSA_Data_Changping.csv
│   ├── PRSA_Data_Dingling.csv
│   ├── PRSA_Data_Dongsi.csv
│   ├── PRSA_Data_Guanyuan.csv
│   ├── PRSA_Data_Gucheng.csv
│   ├── PRSA_Data_Huairou.csv
│   ├── PRSA_Data_Nongzhanguan.csv
│   ├── PRSA_Data_Shunyi.csv
│   ├── PRSA_Data_Tiantan.csv
│   ├── PRSA_Data_Wanliu.csv
│   └── PRSA_Data_Wanshouxigong.csv
├── images/
│   └── station.png            # Gambar yang digunakan di sidebar (diambil dari [Flaticon](https://www.flaticon.com/free-icon/station_1809492))
├── notebook.ipynb             # Notebook Jupyter untuk analisis data
├── LICENSE                    # Lisensi
├── README.md                  # Dokumentasi proyek
└── requirements.txt           # Daftar dependencies yang dibutuhkan
```

## Kontak

Projek ini dibuat oleh **Alif Nurhidayat (KillerKing93)**  
Email: [alifnurhidayatwork@gmail.com](mailto:alifnurhidayatwork@gmail.com)

## Lisensi

Projek ini dilisensikan di bawah [Modified Apache License 2.0 – Non Commercial](LICENSE).  
Lisensi ini mengadaptasi syarat-syarat dari Apache License 2.0 dengan penambahan pembatasan penggunaan untuk keperluan komersial. Penggunaan komersial dari software ini tidak diizinkan kecuali dengan memperoleh lisensi komersial secara terpisah dan membayar loyalty fee sesuai ketentuan. Silakan lihat file LICENSE untuk informasi lengkap.
