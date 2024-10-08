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
- Membuat model machine learning untuk mengidentifikasi pelanggan:<br>
Mengembangkan model machine learning yang dapat memprediksi pelanggan dengan risiko tinggi untuk beralih ke produk lain.<br>
    
#### Solution statements
Berdasarkan permasalahan dan tujuan yang ingin dicapai, maka pada proyek ini solusi dengan tahapan sebagai berikut:
- Melakukan <i>Exploratory Data Analysis</i> pada dataset untuk mengetahui hasil visualisasi data, melihat apakah ada hubungan antar variabel, dan mendapatkan insight dari visualisasi tersebut.
- Mengembangkan model machine learning dengan menggunakan tiga algoritma yakni SVM, Random Forest, dan XGBoost.
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
- Total_Amt_Chng_Q4_Q1: Total jumlah saldo yang berubah dari kuartal 4 ke kuartal 1.  (Integer)
- Total_Trans_Amt: Total jumlah transaksi. (Integer)
- Total_Trans_Ct: Total jumlah transaksi. (Integer)
- Total_Ct_Chng_Q4_Q1: Total frekuensi transaksi yang berubah dari kuartal 4 ke kuartal 1.  (Integer)
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
1. Perbandingan tipe data pelanggan pengguna kartu kredit
![Customer Type](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Customer_Type_Comparation.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa kelas 'Existing Customer' memiliki jumlah data yang jauh lebih banyak dibandingkan dengan kelas 'Attrited Customer' sehingga pada dataset yang digunakan ini terdapat data tak seimbang. Pada tahap data preparation akan dilakukan data balancing dengan metode SMOTE
2. Plot Density Jumlah Hubungan Antara Pengguna dengan Penyedia Jasa Kartu Kredit
![Total Relationship Count](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Total_Relationship_Count.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika jumlah hubungan dengan penyedia jasa kartu kredit kurang dari 2 kali dan terjadi penurunan probabilitas saat pengguna melakukan hubungan dengan penyedia jasa kartu kredit lebih dari 3 kali.
3. Plot Density Jumlah Bulan Pengguna Kartu Kredit Tidak Aktif
![Months Inactive](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Month_Inactive_12.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika 3 sampai 4 bulan tidak aktif menggunakan kartu kredit.
4. Plot Density Jumlah Saldo yang Berputar Pada Pengguna Kartu Kredit
![Revolving Balance](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Total_Revolving_Balance.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika jumlah saldo yang berputar hanya sebesar -500 sampai 600.
5. Plot Density Total Jumlah Saldo saat Transaksi menggunakan Kartu Kredit
![Transaction Amount](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Total_Transfer_Amount.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika jumlah transaksi yang dilakukan menggunakan kartu kredit hanya sekitar 2500 sampai 4000.
6. Plot Density Jumlah Frekuensi Transaksi menggunakan Kartu Kredit
![Transaction Count](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Total_Transfer_Count.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika jumlah frekuensi transaksi yang dilakukan menggunakan kartu kredit hanya sekitar 30 sampai 60 penggunaan.
7. Plot Density Total frekuensi transaksi yang berubah dari kuartal 4 ke kuartal 1 
![Transaction Count Q4 to Q1](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Total_Count_Change_Q4_Q1.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika total frekuensi transaksi yang berubah dari kuartal 4 ke kuartal 1 berada pada skala 0 sampai 0.5 dari total skala 0 sampai 4.0.
8. Plot Density Rata-rata Rasio Penggunaan Kartu Kredit
![Average Utilization Ratio](https://github.com/DestrianaR/Prediksi_Customer_Churn_Pelanggan_Kartu_Kredit/blob/main/Density_Average_Utilization_Ratio.png?raw=true)
Pada hasil visualisasi diatas terlihat bahwa pengguna kartu kredit memiliki probabilitas lebih tinggi menjadi 'Attrited Customer' jika rata-rata rasio penggunaan kartu kredit oleh pengguna adalah skala -0.2 sampai 0.15 dari total skala -0.2 sampai 1.2.

## Data Preparation
Pada tahap ini dilakukan persiapan data sebelum digunakan model dengan tahapan sebagai berikut:
- Split Dataset : Membagi data menjadi X (variabel fitur) dan y (variabel target) dengan proporsi data 8:2.
- Feature Selection : Melakukan pemilihan fitur pada variabel numerik dan variabel kategorik dengan melakukan perhitungan nilai p-value. Variabel fitur yang digunakan pada pembuatan model ini adalah bertipe numerik dan kategorik sehingga untuk data bertipe numerik akan dilakukan pemilihan fitur dengan metode Kendall Tau sedangkan untuk data bertipe kategorik akan digunakan metode Chi-Squared.<br>
    - Numerical Variable <br>
    Kendall Tau adalah metode statistik non-parametrik yang digunakan untuk mengukur derajat korelasi antara variabel fitur bertipe numerik dengan variabel target bertipe kategorik. Kelebihan dari metode ini adalah:
        - Tidak memerlukan asumsi tentang distribusi data
        - Robust terhadap outlier
    - Categoric Variable <br>
    Chi-square test adalah metode statistik yang digunakan untuk menguji hubungan antara dua variabel kategorikal. Kelebihan dari metode ini adalah:
        - Tidak ada asumsi distribusi tertentu yang harus dipenuhi oleh data. 
        - Dapat memberikan hasil yang signifikan dengan sampel yang relatif kecil.
- Handling Outlier : Melakukan pengecekan dan penanganan outlier pada data. Pada tahap ini digunakan metode Winsorizer untuk menangani outlier. Metode Winsorizer merupakan teknik statistik untuk menangani atau mengurangi outlier dari sebaran data. metode tersebut bekerja dengan mengubah nilai outlier menjadi nilai maksimum dan minimum dari sebaran.
- Handling Missing Value : Melakukan pengecekan dan penanganan missing value.
- Feature Scaling : Melakukan perubahan nilai fitur pada data yang digunakan sehingga memiliki skala yang sama.
    - Normal Distribution <br>
        Minmax Scaler digunakan untuk scaling data numerik dengan sebaran skew karena metode ini melakukan perhitungan dengan menggunakan nilai minimum dan maksimum (bukan mean dan media) sehingga metode ini baik digunakan untuk data yang tidak berdistribusi normal.
    - Skew Distribution <br>
        Standard scaler digunakan untuk scaling data numerik dengan sebaran normal karena metode ini melakukan perubahan data sehingga memiliki mean 0 dan standard deviasi 1.
- Feature Encoding : Melakukan encoding atau erubah data pada variabel kategorik menjadi bentuk yang dapat dimengerti oleh algoritma pembelajaran mesin.
    - One-Hot Encoding <br>
    One-Hot Encoding adalah metode encoding yang merubah data kategorik nominal menjadi vektor biner dengan nilai 0 dan 1.
    - Ordinal Encoding <br>
    Ordinal Encoding digunakan untuk mengubah variabel kategorik ordinal menjadi numerik berdasarkan urutan atau peringkat.


## Modeling
Tahapan ini digunakan tiga algoritma yakni SVM, Random Forest, dan XGBoost.
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

### XGBoost
XGBoost merupakan algortima gabungan yang menggunakan pendekatan boosting dimana algoritma ini menggabungkan sejumlah model yang lemah menjadi model yang lebih akurat.
#### Kelebihan
- Memiliki performa yang baik dengan akurasi yang tinggi
#### Kekurangan
- Mudah menjadi overfitting jika parameter tidak ditentukan dengan benar

Model yang dipilih adalah model yang memiliki akurasi yang tinggi pada data train dan data test (tidak overfitting dan underfitting).

## Evaluation
Dengan perbedaan jumlah kelas yang signifikan pada variabel target yang mengindikasikan adanya data tak seimbang, metrik akurasi sudah tidak relevan digunakan untuk evaluasi model ini. Oleh karena itu digunakanlah metrik recall yang lebih sesuai untuk data tak seimbang.

Metrik recall, juga dikenal sebagai sensitivity atau true positive rate (TPR), adalah ukuran evaluasi yang mengukur kemampuan model untuk mengidentifikasi semua kelas positif yang benar dalam suatu dataset. Pada kasus ini kelas positif tersebut adalah kelas yang akan diprediksi yakni 'Attrited Customer'.

Dari hasil model training yang telah dilakukan, didapatkan hasil:
Tabel 2. Hasil Perbandingan Recall Data Train dan Data Test Model SVM, Random Forest, dan XGBoost
|Algoritma |Recall Train	|Recall Test	
---|---|---
SVM	|0.89	|0.82	
Random Forest	|1.00	|0.92	
XGBoost	|0.99	|0.92	

Pada Tabel 2. terlihat bahwa ketiga model mengalami overfitting. Model Random Forest dan XGBoost memiliki nilai yang tidak berbeda jauh namun dilihat dari selisih nilai recall data train dan data test, XGBoost merupakan model yang paling baik dari ketiga model tersebut.

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
Setelah dilakukan pembuatan model machine learning dengan tiga algoritma yang berbeda, diketahui bahwa ketiga model mengalami overfitting dan model XGBoost adalah model dengan tingkat overfitting yang paling rendah terlihat dari selisih nilai recall train dan test yang paling kecil.

**Future Improvement**

Berikut ini adalah beberapa saran yang dapat dilakukan pada pengembangan model machine learning selanjutnya terkait dengan prediksi churn pelanggan kartu kredit.
1. Mengatasi data imbalance
2. Menerapkan Parameter Tuning dengan parameter yang belum digunakan.

## Reference
[1] : D. AL-Najjar, N. Al-Rousan, A. Hazem, "Machine Learning to Develop Credit Card Customer Churn Prediction", <i>J. Theor. Appl.
Electron. Commer. Res.</i>, vol.17, pp. 1529-1542, 2022  [tautan](https://www.mdpi.com/0718-1876/17/4/77)

[2] : H. Tran, N. Le, V. Nguyen, "Customer Churn Prediction in the Banking Sector Using Machine Learning-Based Classification Models", <i>Int. J. Inf. Know. Manag.</i>, vol. 18, pp. 87-105, 2023 [tautan](https://www.informingscience.org/Publications/5086)
