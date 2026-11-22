# Model System Dynamics PM10 ISPU Jakarta (DKI1) - UTS Pemodelan dan Simulasi

## Isi folder
- `Artikel_UTS_SD_ISPU_Jakarta.docx / .pdf` : artikel 2 kolom, Times New Roman 10, spasi 1 (6 halaman)
- `UTS_Simulasi_SD_ISPU_Jakarta.ipynb`      : notebook Google Colab (kalibrasi, validasi, sensitivitas, skenario)
- `sd_model.py`                             : modul model (dipakai notebook dan aplikasi web)
- `app.py`, `requirements.txt`              : aplikasi web Streamlit
- `ispu_dki1.csv`, `params.json`            : dataset dan parameter hasil kalibrasi

## Menjalankan di Google Colab
1. Buka https://colab.research.google.com -> File -> Upload notebook -> pilih `.ipynb`.
2. Jalankan sel dari atas; saat diminta, unggah `ispu_dki1.csv`.

## Deploy web (Streamlit Community Cloud, gratis)
1. Buat repositori GitHub baru, unggah: `app.py`, `sd_model.py`, `requirements.txt`, `params.json`, `ispu_dki1.csv`.
2. Buka https://share.streamlit.io -> New app -> pilih repositori, branch `main`, file utama `app.py` -> Deploy.
3. Salin tautan aplikasi untuk dikumpulkan.

Uji lokal: `pip install -r requirements.txt` lalu `streamlit run app.py`.
