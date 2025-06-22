# Laporan Proyek Predictive Analysis - Telco Customer Churn

## Domain Proyek

Dalam industri telekomunikasi, menjaga loyalitas pelanggan menjadi tantangan utama di tengah ketatnya persaingan antar penyedia layanan. Salah satu indikator penting dalam mengukur loyalitas pelanggan adalah churn rate, yaitu proporsi pelanggan yang berhenti menggunakan layanan dalam periode tertentu. Biaya untuk mendapatkan pelanggan baru jauh lebih tinggi dibandingkan mempertahankan pelanggan yang sudah ada. Oleh karena itu, perusahaan telekomunikasi sangat berkepentingan untuk memprediksi churn sedini mungkin.

Berdasarkan laporan McKinsey (2020), penggunaan machine learning dalam memprediksi churn mampu menurunkan tingkat kehilangan pelanggan hingga 25%. Hal ini membuktikan bahwa analisis prediktif memainkan peran penting dalam strategi retensi pelanggan. Oleh karena itu, proyek ini bertujuan membangun model prediksi churn pelanggan menggunakan algoritma klasifikasi.

**Referensi:**
- McKinsey & Company. (2020). Reducing customer churn through predictive analytics.
- Idris, A., et al. (2020). Predictive analytics for customer churn in telecom using machine learning techniques. IEEE Access.
- Dataset: Kaggle - Telco Customer Churn Dataset

## Business Understanding
Industri telekomunikasi menghadapi tantangan besar dalam mempertahankan pelanggan. Salah satu indikator utama kinerja bisnis adalah tingkat churn—jumlah pelanggan yang berhenti berlangganan dalam periode tertentu. Churn yang tinggi berarti kerugian finansial bagi perusahaan, baik dari kehilangan pendapatan maupun biaya akuisisi pelanggan baru.

Mengidentifikasi pelanggan yang berisiko tinggi untuk berhenti berlangganan memungkinkan perusahaan untuk mengambil langkah preventif seperti memberikan penawaran khusus atau meningkatkan kualitas layanan.

Dengan pendekatan machine learning, perusahaan dapat memprediksi kemungkinan churn berdasarkan pola perilaku pelanggan, sehingga strategi retensi dapat disesuaikan dan dioptimalkan.

### Problem Statements
- Apa saja faktor yang paling memengaruhi kemungkinan pelanggan akan berhenti berlangganan (churn)?
- Bagaimana memprediksi apakah seorang pelanggan akan churn atau tidak berdasarkan data historis pelanggan?

### Goals
- Mengidentifikasi fitur-fitur penting yang berkorelasi tinggi dengan churn pelanggan.
- Membangun model klasifikasi yang mampu memprediksi dengan akurat apakah pelanggan akan churn atau tidak.

### Solution statements
- Membangun beberapa model algoritma untuk melihat model mana yang paling bagus akurasinya.
- Membuat Feature Importance untuk melihat fitur mana saja yang paling berpengaruh terhadap churn.
- Membuat Inference Model dari model yang terindikasi akurasi paling bagus untuk melihat seberapa akurat memprediksi apakah churn atau tidak churn.

## Data Understanding
Dataset yang digunakan adalah **Telco Customer Churn** dari [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data). Dataset ini terdiri dari **7043 baris** dan **21 kolom**, yang mencakup informasi demografis pelanggan, jenis layanan yang digunakan, serta status apakah pelanggan berhenti berlangganan (*churn*) atau tidak.

### Informasi Umum Dataset

- **Total entri**: 7043
- **Jumlah fitur**: 21
- **Target variabel**: `Churn` (Yes / No)

### Variabel pada Dataset Telco Customer Churn

- **customerID**: ID unik pelanggan.
- **Gender**: Jenis kelamin pelanggan (Male, Female).
- **SeniorCitizen**: Apakah pelanggan adalah warga senior (1 = ya, 0 = tidak).
- **Partner**: Apakah pelanggan memiliki pasangan (Yes, No).
- **Dependents**: Apakah pelanggan memiliki tanggungan seperti anak atau keluarga (Yes, No).
- **Tenure**: Lama berlangganan dalam bulan.
- **PhoneService**: Apakah pelanggan menggunakan layanan telepon (Yes, No).
- **MultipleLines**: Apakah pelanggan memiliki beberapa jalur telepon (Yes, No, No phone service).
- **InternetService**: Jenis layanan internet yang digunakan (DSL, Fiber optic, No).
- **OnlineSecurity**: Apakah pelanggan memiliki layanan keamanan online (Yes, No, No internet service).
- **OnlineBackup**: Apakah pelanggan memiliki layanan pencadangan online (Yes, No, No internet service).
- **DeviceProtection**: Apakah pelanggan memiliki perlindungan perangkat (Yes, No, No internet service).
- **TechSupport**: Apakah pelanggan memiliki dukungan teknis (Yes, No, No internet service).
- **StreamingTV**: Apakah pelanggan menggunakan layanan TV streaming (Yes, No, No internet service).
- **StreamingMovies**: Apakah pelanggan menggunakan layanan film streaming (Yes, No, No internet service).
- **Contract**: Jenis kontrak langganan (Month-to-month, One year, Two year).
- **PaperlessBilling**: Apakah pelanggan menggunakan tagihan digital (Yes, No).
- **PaymentMethod**: Metode pembayaran yang digunakan (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- **MonthlyCharges**: Total biaya bulanan yang dibebankan ke pelanggan.
- **TotalCharges**: Total biaya selama pelanggan berlangganan.
- **Churn**: Apakah pelanggan berhenti berlangganan (Yes, No).

### Cek Missing Values

Pemeriksaan nilai hilang dilakukan menggunakan fungsi `df.isnull().sum()` dan `df.info()`. Secara eksplisit, tidak ditemukan nilai `NaN`. Namun, terdapat **nilai kosong tersembunyi** dalam kolom `TotalCharges` dalam bentuk string kosong (`''`), yang perlu ditangani.

- **Kolom TotalCharges**: Ditemukan sekitar **11 baris** dengan nilai kosong.
  - **Penanganan**: Nilai kosong akan diisi dengan 0 sebagai indikasi total tagihan awal.
  - **Konversi tipe**: `TotalCharges` perlu diubah ke tipe numerik (`float`).

### Cek Duplikasi

Pemeriksaan menggunakan `df.duplicated().sum()` menunjukkan bahwa **tidak terdapat baris duplikat** pada dataset ini.

### Statistik Deskriptif

Berikut adalah ringkasan statistik untuk kolom numerik:

| Kolom           | Count  | Mean   | Std Dev | Min   | 25%   | 50%   | 75%   | Max     |
|-----------------|--------|--------|---------|-------|-------|-------|-------|---------|
| `Tenure`        | 7043   | 32.37  | 24.56   | 0.00  | 9.00  | 29.00 | 55.00 | 72.00   |
| `MonthlyCharges`| 7043   | 64.76  | 30.09   | 18.25 | 35.50 | 70.35 | 89.85 | 118.75  |
| `TotalCharges`  | 7043   | 2279.73| 2266.79 | 0.00  | 398.55| 1394.55| 3786.60| 8684.80 |

**Catatan penting**:
- Nilai minimum `TotalCharges` adalah **0**, yang tampaknya tidak sebanding dengan `tenure > 0`, sehingga perlu dicurigai sebagai **entri anomali** atau pelanggan yang baru saja mendaftar.
- Rentang pada `MonthlyCharges` cukup lebar, menunjukkan adanya variasi besar dalam jenis layanan pelanggan.
- `Tenure` maksimal adalah 72 bulan (6 tahun), dan nilai tengahnya (median) adalah 29 bulan, menunjukkan sebagian besar pelanggan merupakan pelanggan jangka menengah.

Statistik ini membantu memahami sebaran data numerik dan menjadi dasar dalam mendeteksi outlier serta melakukan transformasi atau normalisasi jika diperlukan.


### Tipe Data

Berdasarkan hasil `df.info()`, hampir semua kolom bertipe `object` kecuali:

- `SeniorCitizen` (integer 0 atau 1)
- `tenure` (integer)
- `MonthlyCharges` (float)

Rencana konversi tipe data:
- Seluruh kolom bertipe `object` selain `customerID` dan `TotalCharges` akan dikonversi menjadi tipe `category`.
- `TotalCharges` akan dikonversi ke `float` setelah menangani nilai kosong.

### Distribusi Target (Churn)

Distribusi dari variabel target `Churn` adalah sebagai berikut:

- **Tidak churn (No)**: 73.46%
- **Churn (Yes)**: 26.54%

Distribusi yang tidak seimbang ini mengindikasikan adanya **class imbalance** yang perlu diperhatikan pada saat modelling.

## Data Preparation
Berikut adalah tahapan persiapan data yang dilakukan, sesuai urutan dalam notebook:

1. **Penanganan Missing Value**:
   - Kolom `TotalCharges ` diisi dengan nilai 0.
   - **Alasan**: Data `TotalCharges ` berisikan total tagihan, pada kasus ini total tagihannya kosong diisi dengan blank space, itu menandakan bahwa pelanggan baru bergabung dan belum memiliki total tagihan, maka dari itu diisi dengan 0.

2. **Pemisahan Fitur dan Target**:
   - Fitur (X) adalah semua kolom kecuali `customerID` dan `churn`, dan target (y) adalah `churn`.
   - **Alasan**: Karena kita ingin memisahkan fitur (X) dan target (y), maka kolom `churn` sebagai target perlu dipisahkan dari fitur, dan kolom `customerID` dihapus karena hanya berfungsi sebagai identitas unik yang tidak memiliki kontribusi dalam proses prediksi..

3. **Encoding Fitur Kategori**:
   - Data selain `customerID`, `SeniorCitizen`, `Tenure`, dan `MonthlyCharges` akan di encode menggunakan `OneHotEncoder`.
   - **Alasan**: One-hot encoding digunakan untuk mengubah fitur kategorikal menjadi bentuk numerik agar dapat diproses oleh algoritma machine learning, karena sebagian besar algoritma tidak dapat menangani data dalam bentuk string.

4. **Normalisasi Fitur Numerik**:
   - Fitur numerik diskalakan menggunakan `StandardScaler` untuk menstandarisasi skala.
   - **Alasan**: Normalisasi menggunakan `StandardScaler` dilakukan untuk menyamakan skala semua fitur numerik, sehingga setiap fitur memiliki rata-rata 0 dan standar deviasi 1. Hal ini penting agar algoritma machine learning yang sensitif terhadap skala (seperti Logistic Regression atau KNN) tidak memihak fitur dengan nilai lebih besar.

5. **Pembagian Data**:
   - Data dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split`.
   - **Alasan**: Untuk mengevaluasi performa model pada data yang belum dilihat.

6. **Jumlah Data Split**:
   - Melihat jumlah dari setiap `X_train` dan `X_test`.
   - **Alasan**: Memastikan jumlah data yang di split sesuai dengan total pada dataset.

## Modeling

Tahapan ini membahas mengenai pemilihan dan pelatihan beberapa algoritma machine learning untuk menyelesaikan permasalahan prediksi *customer churn*. Delapan algoritma klasifikasi digunakan untuk mengevaluasi performa dan memilih model terbaik berdasarkan hasil evaluasi pada data uji.

### Model yang Digunakan dan Parameternya:

1. **K-Nearest Neighbors (KNN)**  
   - **Parameter**: `n_neighbors=5`  
   - **Prinsip kerja**: KNN mengklasifikasikan data baru berdasarkan mayoritas kelas dari *k* tetangga terdekatnya. Jarak antar data biasanya dihitung menggunakan Euclidean distance.  
   - **Kelebihan**: Sederhana dan efektif untuk dataset kecil.  
   - **Kekurangan**: Kurang efisien untuk dataset besar dan sensitif terhadap skala fitur.

2. **Random Forest**  
   - **Parameter**: `n_estimators=100`, `random_state=42`  
   - **Prinsip kerja**: Random Forest membangun sejumlah pohon keputusan pada subset acak dari data, kemudian mengambil voting mayoritas dari semua pohon untuk menentukan prediksi akhir.  
   - **Kelebihan**: Akurasi tinggi, tahan terhadap overfitting, dan dapat mengukur feature importance.  
   - **Kekurangan**: Lebih kompleks dan memerlukan lebih banyak memori.

3. **Decision Tree**  
   - **Parameter**: `random_state=42`  
   - **Prinsip kerja**: Decision Tree membagi data secara rekursif berdasarkan fitur yang memberikan informasi paling tinggi hingga mencapai kondisi terminal (daun).  
   - **Kelebihan**: Mudah dipahami dan divisualisasikan.  
   - **Kekurangan**: Cenderung overfitting pada data training jika tidak dilakukan pruning.

4. **Logistic Regression**  
   - **Parameter**: `random_state=42`  
   - **Prinsip kerja**: Logistic Regression memodelkan hubungan antara variabel input dan probabilitas keluaran biner dengan menggunakan fungsi logistik.  
   - **Kelebihan**: Mudah diinterpretasi dan menjadi baseline yang baik untuk klasifikasi biner.  
   - **Kekurangan**: Mengasumsikan hubungan linear antar fitur.

5. **Naive Bayes (GaussianNB)**  
   - **Parameter**: default (`var_smoothing=1e-9`)  
   - **Prinsip kerja**: Gaussian Naive Bayes mengasumsikan bahwa fitur mengikuti distribusi normal dan saling independen. Prediksi didasarkan pada probabilitas posterior menggunakan Teorema Bayes.  
   - **Kelebihan**: Cepat dan efisien untuk dataset dengan fitur numerik.  
   - **Kekurangan**: Mengasumsikan independensi antar fitur, yang jarang terjadi dalam kenyataan.

6. **Support Vector Classifier (SVC)**  
   - **Parameter**: `random_state=42`  
   - **Prinsip kerja**: SVC mencari hyperplane terbaik yang memisahkan dua kelas dengan margin maksimum. Cocok untuk dataset berdimensi tinggi dan dapat menggunakan kernel untuk data yang tidak linear.  
   - **Kelebihan**: Akurat untuk dataset kecil hingga menengah, terutama yang berdimensi tinggi.  
   - **Kekurangan**: Kurang efisien untuk dataset besar dan sensitif terhadap parameter.

7. **AdaBoost Classifier**  
   - **Parameter**: `random_state=42`  
   - **Prinsip kerja**: AdaBoost bekerja dengan membangun model secara bertahap, di mana setiap model berikutnya fokus pada data yang salah diklasifikasikan oleh model sebelumnya.  
   - **Kelebihan**: Dapat meningkatkan akurasi model lemah seperti decision tree.  
   - **Kekurangan**: Sensitif terhadap noise dan outlier.

8. **Gradient Boosting Classifier**  
   - **Parameter**: `random_state=42`  
   - **Prinsip kerja**: Gradient Boosting membangun model prediktif secara bertahap, dengan setiap model baru mengoreksi kesalahan dari model sebelumnya berdasarkan gradien dari fungsi loss.  
   - **Kelebihan**: Performa tinggi dalam berbagai tugas klasifikasi.  
   - **Kekurangan**: Pelatihan bisa lama dan membutuhkan tuning parameter yang teliti.


### Proses Pemodelan

Semua model dilatih menggunakan data training yang telah diproses sebelumnya (melalui encoding, normalisasi, dan pembagian 80:20). Tidak semua model menggunakan parameter tuning pada tahap ini, karena tujuan awal adalah membandingkan performa antar algoritma secara default.

### Pemilihan Model Terbaik

Setelah dilakukan evaluasi terhadap semua model (pada bagian *Evaluation*), model terbaik dipilih berdasarkan kombinasi metrik evaluasi seperti **accuracy**, **precision**, **recall**, dan **F1-score**, khususnya untuk kelas minoritas (churn = 1). Model yang dipilih adalah yang memberikan **keseimbangan terbaik antara sensitivitas dan presisi**.

**Model Logistic Regression** menunjukkan performa terbaik dalam mendeteksi pelanggan yang berisiko churn, dengan skor F1 tertinggi dan recall yang baik. Oleh karena itu, model ini dipilih sebagai model akhir.

## Evaluation

Pada tahap evaluasi ini, kami menggunakan beberapa metrik utama untuk menilai performa model klasifikasi dalam memprediksi churn pelanggan. Metrik yang digunakan meliputi:

- **Accuracy**  
  Mengukur proporsi prediksi yang benar dari seluruh data.  
  Formula:  

         Accuracy = (TP + TN) / (TP + TN + FP + FN)
  
  Kelemahan akurasi adalah kurang sensitif terhadap ketidakseimbangan kelas.

- **Precision**  
    Mengukur proporsi prediksi positif yang benar-benar positif.  
    Formula: 
    
        Precision = TP / (TP + FP)

  Precision penting ketika biaya kesalahan positif (false positive) tinggi.

- **Recall (Sensitivity)**  
    Mengukur proporsi data positif yang berhasil dideteksi.  
    Formula:  

        Recall = TP / (TP + FN)

  Recall penting ketika ingin meminimalkan kesalahan negatif (false negative).

- **F1-Score**  
    Merupakan harmonic mean dari precision dan recall, memberikan keseimbangan antara keduanya.  
    Formula:  

        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

### Hasil Evaluasi

Dari hasil evaluasi delapan model yang diuji, diperoleh metrik sebagai berikut:

| Model                | Accuracy | F1-Score | Precision | Recall |
|----------------------|----------|----------|-----------|--------|
| Logistic Regression  | 0.82     | 0.82     | 0.82      | 0.82   |
| SVC                  | 0.81     | 0.80     | 0.80      | 0.81   |
| Gradient Boosting    | 0.81     | 0.80     | 0.80      | 0.81   |
| AdaBoost             | 0.81     | 0.80     | 0.80      | 0.81   |
| Random Forest        | 0.79     | 0.78     | 0.78      | 0.79   |
| K-Nearest Neighbors  | 0.77     | 0.77     | 0.77      | 0.77   |
| Decision Tree        | 0.72     | 0.72     | 0.72      | 0.72   |
| Naive Bayes          | 0.70     | 0.71     | 0.80      | 0.70   |

Berdasarkan hasil tersebut, model **Logistic Regression** menunjukkan performa terbaik secara keseluruhan dengan nilai akurasi tertinggi sebesar **82%**, F1-Score **82%**, precision **82%**, dan recall **82%**. Hal ini menunjukkan bahwa model ini mampu memprediksi pelanggan yang melakukan churn dengan baik dan seimbang, baik dari segi ketepatan prediksi maupun kemampuan mendeteksi pelanggan yang benar-benar churn.

### Analisis Feature Importance

Berdasarkan hasil feature importance dari model yang digunakan **Logistic Regression** dan **Random Forest**, variabel-variabel yang paling berpengaruh terhadap prediksi churn adalah sebagai berikut:

| No | Feature                |
|----|------------------------|
| 0  | InternetService_DSL     |
| 1  | Tenure                 |
| 2  | Contract_Month-to-month |
| 3  | OnlineSecurity_No       |
| 4  | MonthlyCharges         |
| 5  | Contract_Two year       |
| 6  | TotalCharges           |

**Penjelasan:**

- **InternetService_DSL**: Pelanggan dengan layanan internet DSL memiliki kecenderungan tertentu terhadap churn dibandingkan layanan lainnya.
- **Tenure**: Lama berlangganan pelanggan sangat berpengaruh; pelanggan dengan tenure pendek lebih mungkin churn.
- **Contract_Month-to-month**: Pelanggan dengan kontrak bulanan (month-to-month) lebih rentan churn dibandingkan kontrak jangka panjang.
- **OnlineSecurity_No**: Pelanggan yang tidak memiliki layanan keamanan online berpotensi lebih besar untuk churn.
- **MonthlyCharges**: Biaya bulanan yang tinggi dapat meningkatkan risiko churn.
- **Contract_Two year**: Kontrak dua tahun cenderung mengurangi churn, menandakan loyalitas yang lebih tinggi.
- **TotalCharges**: Total biaya yang dibayar pelanggan juga memberikan indikasi churn, di mana pelanggan dengan total biaya rendah biasanya baru berlangganan dan lebih rentan churn.

Visualisasi feature importance ini dapat membantu bisnis dalam mengidentifikasi faktor-faktor utama yang harus diperhatikan untuk strategi retensi pelanggan.

### Inference pada Data Baru

Setelah model berhasil dilatih dan dievaluasi, selanjutnya dilakukan prediksi (inference) pada data pelanggan baru yang belum pernah dilihat oleh model.

#### Data Baru

Data baru yang akan diprediksi memiliki 4 contoh pelanggan dengan berbagai karakteristik sebagai berikut:

| Gender | SeniorCitizen | Partner | Dependents | Tenure | PhoneService | MultipleLines | InternetService | OnlineSecurity | OnlineBackup | DeviceProtection | TechSupport | StreamingTV | StreamingMovies | Contract       | PaperlessBilling | PaymentMethod           | MonthlyCharges | TotalCharges |
|--------|---------------|---------|------------|--------|--------------|---------------|-----------------|----------------|--------------|------------------|-------------|-------------|-----------------|----------------|------------------|------------------------|----------------|--------------|
| Male   | 0             | Yes     | No         | 12     | Yes          | No            | Fiber optic     | No             | Yes          | No               | No          | Yes         | No              | Month-to-month | Yes              | Electronic check       | 75.5           | 900.5        |
| Female | 0             | Yes     | Yes        | 72     | Yes          | Yes           | DSL             | Yes            | Yes          | Yes              | Yes         | Yes         | Yes             | Two year       | No               | Credit card (automatic) | 65.0           | 4500.0       |
| Female | 0             | Yes     | Yes        | 70     | Yes          | Yes           | DSL             | Yes            | Yes          | Yes              | Yes         | Yes         | Yes             | Two year       | No               | Bank transfer (automatic) | 55.2         | 3800.5       |
| Male   | 1             | No      | No         | 1      | Yes          | No            | Fiber optic     | No             | No           | No               | No          | Yes         | Yes             | Month-to-month | Yes              | Electronic check       | 95.7           | 95.7         |

#### Proses Prediksi

1. Data pelanggan di-encode menggunakan one-hot encoding agar sesuai dengan fitur yang digunakan saat pelatihan model.
2. Data di-reindex agar kolomnya sesuai dengan kolom fitur pelatihan (`X_columns`).
3. Fitur numerik distandarisasi menggunakan `scaler` yang sama dengan saat pelatihan.
4. Model Logistic Regression (`lr`) melakukan prediksi churn untuk setiap pelanggan.

#### Kode Prediksi

```python
# One-hot encode data baru
new_data_encoded = pd.get_dummies(sample_data)

# Reindex agar kolom sesuai dengan X_columns saat training
new_data_encoded = new_data_encoded.reindex(columns=X_columns, fill_value=0)

# Lakukan scaling hanya untuk kolom numerik
new_data_encoded[numerical_features] = scaler.transform(new_data_encoded[numerical_features])

# Prediksi semua baris
pred = lr.predict(new_data_encoded)

# Interpretasi hasil untuk semua data
for i, p in enumerate(pred):
    print(f"Data ke-{i+1} prediksi churn:", "Ya (Churn)" if p == 1 else "Tidak (Tidak Churn)")
```

### Interpretasi

Model yang dibangun berhasil memenuhi tujuan bisnis dalam memprediksi churn pelanggan dengan akurasi yang tinggi dan metrik evaluasi lain yang seimbang, seperti precision dan recall. Hal ini menunjukkan model dapat membedakan pelanggan yang akan berhenti berlangganan dan yang tetap dengan tingkat kesalahan yang rendah, sehingga mengurangi risiko mengambil tindakan yang salah.

Analisis *Feature Importance* mengungkapkan bahwa faktor-faktor utama yang mempengaruhi churn adalah lama masa berlangganan (`tenure`), jenis kontrak pelanggan (`Contract_Month-to-month` atau `Contract_Two_year`), total biaya yang sudah dibayar (`TotalCharges`), serta jenis layanan internet yang digunakan (`InternetService_DSL`). Insight ini sangat berharga bagi bisnis karena dapat digunakan untuk merancang strategi retensi yang lebih fokus, seperti memberikan insentif kepada pelanggan dengan kontrak bulanan atau pelanggan dengan masa berlangganan rendah.

Dengan demikian, model dan analisis yang dilakukan sudah menjawab *problem statement* utama, yaitu mengetahui faktor-faktor pengaruh churn dan memprediksi churn secara akurat. Model ini juga berhasil mencapai *goals* bisnis yaitu membangun model klasifikasi yang handal serta mengidentifikasi fitur-fitur penting. Solusi yang diterapkan memberikan dampak signifikan dengan memungkinkan perusahaan untuk mengambil keputusan berbasis data yang lebih efektif dalam mengurangi churn pelanggan.

### Kesimpulan

Model **Logistic Regression** berhasil menjadi pilihan terbaik karena memberikan performa yang optimal dalam memprediksi churn pelanggan berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score. Analisis fitur mengonfirmasi bahwa variabel `tenure`, `Contract_Month-to-month`, `Contract_Two_year`, `TotalCharges`, dan `InternetService_DSL` adalah faktor kunci yang memengaruhi keputusan pelanggan untuk churn.

Dengan kemampuan prediksi yang akurat dan pemahaman mendalam terhadap faktor-faktor yang berpengaruh, model ini memberikan dasar yang kuat bagi perusahaan untuk merancang strategi retensi pelanggan yang tepat sasaran. Dengan demikian, model ini tidak hanya menjawab problem statement dan goals yang telah ditetapkan, tetapi juga memberikan solusi praktis yang berdampak nyata dalam upaya menurunkan tingkat churn dan meningkatkan loyalitas pelanggan.

Ke depannya, perusahaan dapat memanfaatkan insight ini untuk mengembangkan program insentif, memperbaiki layanan pelanggan, dan meningkatkan kepuasan pelanggan guna mempertahankan pangsa pasar secara lebih efektif.
