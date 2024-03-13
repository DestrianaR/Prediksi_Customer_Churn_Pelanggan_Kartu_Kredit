# Laporan Proyek Machine Learning: Prediksi Churn Pelanggan Kartu Kredit - Destriana Ramadani

## Domain Proyek

Saat ini, pasar menjadi dinamis dan sangat kompetitif karena banyaknya jumlah penyedia layanan jasa, dimana salah satunya adalah jasa perbankan. Tantangan bagi sektor ini adalah perubahan prilaku pelanggan dimana pelanggan adalah inti dari semua industri. Pelanggan jangka panjang secara langsung terhubung dengan keuntungan perusahaan, sehingga bank harus menghindari kehilangan pelanggan.

Customer Churn adalah istilah bisnis dimana pelanggan tidak puas sengan layanan atau produk yang diberikan sehingga terjadi atrisi (attrition) atau dengan kata lain pelanggan berhenti terhubung dan menggunakan jasa bisnis suatu perusahaan. Biasanya, mudah untuk menemukan dari analisis bahwa sebagian besar keuntungan perusahaan dikontribusikan oleh pelanggan tetap dan lebih banyak biaya yang dikeluarkan untuk menarik pelanggan baru daripada mempertahankan pelanggan lama. Oleh karena itu, menemukan churner dapat membantu perusahaan mempertahankan pelanggan mereka dan mempertahankan hubungan dengan pelanggan yang sudah ada menjadi lebih penting.

### Manfaat dan Dampak
Perpindahan pelanggan merugikan bisnis karena dapat mengakibatkan kerugian premi yang besar, penurunan margin keuntungan, dan kemungkinan kehilangan bisnis rujukan dari pelanggan setia. Untuk membatasi hal ini, perusahaan harus mengantisipasi pelanggan tertentu yang berisiko pergi dengan menyesuaikan strategi mereka, seperti meningkatkan kualitas produk dan layanan atau meningkatkan manfaatnya. Oleh karena itu, sangat penting untuk membuat model prediktif yang dapat membantu retensi pelanggan.
  

### Article and Reference
- [Machine Learning to Develop Credit Card Customer Churn Prediction](https://www.mdpi.com/0718-1876/17/4/77) 
- [Customer Churn Prediction in the Banking Sector Using Machine Learning-Based Classification Models](https://www.informingscience.org/Publications/5086)

## Business Understanding
### Problem Statements
- Apa saja faktor-faktor yang mempengaruhi churn pelanggan ?
- Bagaimana mengidentifikasi pelanggan yang memiliki risiko tinggi untuk beralih menggunakan produk lain?

### Goals
- Melakukan uji korelasi untuk mengetahui faktor-faktor yang mempengaruhi churn pelanggan:<br>
Menganalisis data untuk menentukan hubungan antara berbagai faktor dan kemudian melakukan uji korelasi statistik untuk mengidentifikasi hubungan yang signifikan.<br>
-Membuat model machine learning untuk mengidentifikasi pelanggan:<br>
Mengembangkan model machine learning yang dapat memprediksi pelanggan dengan risiko tinggi untuk beralih ke produk lain.<br>
    
#### Solution statements
Berdasarkan permasalahan dan tujuan yang ingin dicapai, maka pada proyek ini solusi dengan tahapan sebagai berikut:
- Melakukan <i>Exploratory Data Analysis</i> pada dataset untuk mengetahui hasil visualisasi data, melihat apakah ada hubungan antar variabel, dan mendapatkan insight dari visualisasi tersebut.
- Mengembangkan model machine learning dengan menggunakan dua algoritma yakni SVM, Random Forest, dan XGBoost.
    - SVM (Support Vector Machine) : Algoritma yang mencari daerah keputusan terbaik (hyperplane) yang memisahkan dua kelas dalam ruang fitur.
        - Kelebihan : Efektif dalam ruang fitur berdimensi tinggi
        - Kelemahan : Rentan terhadap overfitting jika parameter tidak diatur dengan baik
    - Random Forest : Algoritma pembelajaran gabungan yang terdiri dari decision tree yang tiap tree-nya dihasilakan secara acak dan independen dimana outputnya adalah kombinasi dari prediksi dari masing-masing tree.
        - Kelebihan : Dapat menangani overfitting yang sering muncul dalam decision tree
        - Kelemahan : Membutuhkan lebih banyak sumber daya komputasi. 
    - XGBoost adalah algoritma gabungan yang menggunakan pendekatan boosting. Menggabungkan sejumlah besar model yang lemah untuk menciptakan model yang kuat.
        - Kelebihan : Kinerja tinggi dan akurasi yang baik
        - Kelemahan : Rentan terhadap overfitting jika parameter tidak diatur dengan benar.
- Melakukan evaluasi berdasarkan metriks evaluasi dari model prediksi yang telah dikembangakan. Karena masalah pada projek ini adalah masalah klasifikasi dengan tujuan untuk memprediksi pelanggan dengan potensi churn yang tinggi (False Negative) maka metrics utama yang akan digunakan adalah recall.


## Data Understanding
Data yang digunakan pada proyek ini adalah Data Credit Card Churn Prediction yang dapat diunduh pada link berikut  : [Kaggle Credit Card Churn Prediction](https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn).

### Variabel-variabel pada Kaggle Credit Card Churn Prediction dataset adalah sebagai berikut:
- Attrition_Flag: Tanda yang menunjukkan apakah pelanggan telah meninggalkan. (Boolean)
- Customer_Age:  Usia pelanggan. (Integer)
- Gender: Jenis kelamin pelanggan. (String)
- Dependent_count: Jumlah tanggungan yang dimiliki pelanggan.  (Integer)
- Education_Level: Tingkat pendidikan pelanggan.  (String)
- Marital_Status: Status perkawinan pelanggan. (String)
- Income_Category: Kategori pendapatan pelanggan. (String)
- Card_Category: Jenis kartu yang dimiliki oleh pelanggan. (String)
- Months_on_book: Berapa lama pelanggan telah terdaftar.  (Integer)
- Total_Relationship_Count: Total jumlah hubungan yang dimiliki pelanggan dengan penyedia kartu kredit. (Integer)
- Months_Inactive_12_mon: Jumlah bulan pelanggan tidak aktif dalam dua belas bulan terakhir. (Integer)
- Contacts_Count_12_mon:  Jumlah kontak yang dilakukan pelanggan dalam dua belas bulan terakhir. (Integer)
- Credit_Limit: Batas kredit pelanggan.. (Integer)
- Total_Revolving_Bal: Total saldo yang berputar dari pelanggan. (Integer)
- Avg_Open_To_Buy: Rasio rata-rata pembelian yang terbuka bagi pelanggan. (Integer)
- Total_Amt_Chng_Q4_Q1: Total jumlah yang berubah dari kuartal 4 ke kuartal 1.  (Integer)
- Total_Trans_Amt: Total jumlah transaksi. (Integer)
- Total_Trans_Ct: Total jumlah transaksi. (Integer)
- Total_Ct_Chng_Q4_Q1: Total jumlah yang berubah dari kuartal 4 ke kuartal 1.  (Integer)
- Avg_Utilization_Ratio: Rasio penggunaan rata-rata pelanggan. (Integer)

### Dataset Overview
Tabel 1. Subset Dataset Informasi Pengguna Kartu Kredit 
| CLIENTNUM | Attrition_Flag | Customer_Age | Gender | Dependent_count | Education_Level | Marital_Status | Income_Category | Card_Category | 
|-----------|----------------|--------------|--------|-----------------|-----------------|----------------|-----------------|---------------|
| 768805383         | Existing Customer | 45         | M      | 3               | High School     | Married        | $60K - $80K     | Blue          |
| 818770008         | Existing Customer | 49         | F      | 5               | Graduate     | Single        | Less than $40K	     | Blue          |
| 713982108         | Existing Customer | 51         | M      | 3               | Graduate     | Married        | $80K - $120K	     | Blue          |
| 769911858         | Existing Customer | 40         | F      | 4               | High School     | Unknown        | Less than $40K	     | Blue          |
| 709106358         | Existing Customer | 40         | M      | 3               | Uneducated     | Married        | $60K - $80K	     | Blue          |

Tabel 1. menunjukkan subset dataset informasi pengguna kartu kredit sebanyak 5 baris dan 9 kolom dari total dataset awal sebanyak 10127 baris dan 23 kolom. Kemudian dilakukan penghapusan 3 kolom yang tidak digunakan pada model sehingga hanya digunakan 20 kolom. Total data numerik pada data yang digunakan sebanyak 14 dan kolom kategorik sebanyak 6.
 
## Exploratory Data Analysis:
Berikut ini adalah hasil exploratory analysis:

## Data Preparation
Pada tahap ini dilakukan persiapan data sebelum digunakan model dengan tahapan sebagai berikut:
- Split Dataset : Membagi data menjadi X (variabel fitur) dan y (variabel target). Kemudian dari kedua variabel tersebut dibagi menjadi X_train, X_test, y_train, dan y_test dengan proporsi 9:1.
- Split Numerical and Categorical Columns : Membagi data sesuai tipe data agar memudahkan proses persiapan data selanjutnya
- Handling Outlier : Melakukan pengecekan dan penanganan outlier pada data sebaran normal dan sebaran skew dengan metode Winsorizer yakni mengubah data outlier menjadi data maksimum dan minimun dalam sebaran data sehingga tidak ada data yang dibuang.
- Handling Missing Value : Melakukan pengecekan dan penanganan missing value.
- Feature Selection : Melakukan pemilihan fitur pada variabel numerik dan variabel kategorik dengan melakukan perhitungan nilai p-value.
- Cardinality : Melakukan pengecekan kardinality pada data kategorik untuk mengetahui jumlah kategori pada masing-masing variabel. Jika variabel memiliki kardinalitas yang tinggi, maka akan dilakukan data manipulasi untuk menurunkan kardinalitas.
- Feature Scaling : Melakukan scaling pada variabel numerik dengan metode Standard Scaler agar sebaran data menjadi normal.
- Feature Encoding : Melakukan encoding atau merubah data kategorik menjadi data numerik dengan dua metode yakni One Hot Encoding untuk data nominal dan Ordinal Encoding untuk data ordinal.
- Label Encoding : Melakukan encoding pada variabel target menjadi data numerik

## Modeling
Tahapan ini digunakan dua algoritma yakni SVM dan Random Forest.
### SVM
Support Vector Machine (SVM) merupakan algoritma yang berkerja untuk menemukan batas keputusan yang memaksimalkan jarak dari titik data terdekat pada semua kelas data.
#### Kelebihan
- Efektif saat jumlah fitur lebih besar dari jumlah sampel
- Efektif untuk dataset yang kecil
#### Kekurangan
- Tidak direkomendasikan untuk dataset yang besar.
- Sensitif terhadap perubahan dan outlier.

### Random Forest
Random Forest merupakan algoritma gabungan yang terdiri dari kombinasi algoritma decision tree.
#### Kelebihan
- Efektif untuk menangani data yang besar dengan dimensi yang tinggi.
- Tidak sensitif terhadap outlier dan missing value.
#### Kekurangan
- Model yang kompleks sehingga waktu komputasi yang dibutuhkan cukup lama.
Model yang dipilih adalah model yang memiliki akurasi yang tinggi pada data train dan data test (tidak overfitting dan underfitting).

## Evaluation
Dari hasil model training yang telah dilakukan, didapatkan hasil:
- SVM
    - Akurasi data train : 0.96 
    - Akurasi data test : 0.94
- Random Forest
    - Akurasi data train : 1.00 
    - Akurasi data test : 0.95

Terlihat bahwa akurasi data train Random Forest lebih besar dibandingkan dengan SVM, namun terdapat selisih yang cukup besar jika dibandingkan dengan akurasi data testnya, sehingga terdapat kemungkinan overfitting pada model tersebut. Untuk selanjutnya akan digunakan model svm untuk menjelaskan problem statement pada proyek ini.

### Interpretasi Metrik 
Goals atau tujuan dari proyek ini adalah untuk memprediksi pelanggan dengan risiko tinggi untuk beralih ke produk lain sehingga metrik yang digunakan adalah precision dari hasil prediksi oleh model terhadap 'Attrition Customer'.

![alt text](output.png)

Diketahui label 0 = 'Existing Customer' dan 1 = 'Attrition Customer'.
Dari hasil pengecekan prediksi menggunakan data testing didapatkan nilai precison label 1 sebesar 0.98 yang artinya <b>Model memprediksi dengan benar 98% pelanggan yang memiliki potensi beralih ke produk perbankan lain. </b>

### Faktor-Faktor yang mempengaruhi Churn pelanggan
Setelah dilakukan uji statistik pada tahap feature selection, berikut ini adalah variabel yang mempengaruh Churn pelanggan yang juga digunakan sebagai variabel feature pada model
- Gender
- Income_Category
- Dependent_count
- Total_Relationship_Count
- Months_Inactive_12_mon 
- Contacts_Count_12_mon 
- Credit_Limit
- Total_Revolving_Bal 
- Avg_Open_To_Buy 
- Total_Amt_Chng_Q4_Q1 
- Total_Trans_Amt 
- Total_Trans_Ct 
- Total_Ct_Chng_Q4_Q1 
- Avg_Utilization_Ratio 

## Conclusion
Setelah dilakukan pembuatan model machine learning didapatkan model yang cukup baik dalam memprediksi churn pelanggan. Dari dua algoritma yang digunakan dipilih algoritma SVM dengan akurasi model sebesar 0.94. 

Kemudian dilakukan pengecekan variabel yang mempengaruhi churn pelanggan. Dari hasil tersebut dapat memberikan informasi karakteristik pelanggan yang berpotensi churn dan membantu dalam membuat strategi bisnis kedepannya.

## Reference
[1] : D. AL-Najjar, N. Al-Rousan, A. Hazem, "Machine Learning to Develop Credit Card Customer Churn Prediction", <i>J. Theor. Appl.
Electron. Commer. Res.</i>, vol.17, pp. 1529-1542, 2022  [tautan](https://www.mdpi.com/0718-1876/17/4/77)

[2] : H. Tran, N. Le, V. Nguyen, "Customer Churn Prediction in the Banking Sector Using Machine Learning-Based Classification Models", <i>Int. J. Inf. Know. Manag.</i>, vol. 18, pp. 87-105, 2023 [tautan](https://www.informingscience.org/Publications/5086)