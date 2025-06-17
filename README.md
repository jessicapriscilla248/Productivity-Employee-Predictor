
# Employee Productivity Predictor - Garmentivity 
Aplikasi machine learning untuk memprediksi apakah seorang karyawan masuk ke dalam kategori produktif atau tidak menggunakan pengembangan model **CatBoostClassifier**.
---

## Instalasi & Persiapan
### 1. Clone Repository
```bash
git clone https://github.com/jessicapriscilla248/Productivity-Employee-Predictor.git
cd Productivity-Employee-Predictor
```

### 2. Buat Virtual Environment 
```bash
python -m venv env
source env/bin/activate      # MacOS/Linux
env\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Aplikasi

### Jalankan dengan Streamlit
```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser di `http://localhost:8501`.
---

## 📥 Input Data
- User dapat input individual data lewat form.
- User dapat upload file `.csv` dengan format kolom sesuai template.

---

## 📊 Hasil Prediksi
- Model akan menampilkan apakah karyawan tersebut produktif atau tidak.
- Diberikan rekomendasi untuk HR untuk step selanjutnya berdasarkan kinerja karyawannya
---

## 🧠 Model yang Digunakan
- `CatBoostClassifier` karena model ini mendukung fitur kategorikal secara native dan efisien.
---

## 👩‍💻 Dibuat oleh
> Jessica Priscilla Immanuel, Kevin Erdianto Simon, Meghan Hillary Mardjohan
