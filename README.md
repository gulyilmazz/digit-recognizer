***📄 digit-recognizer


🎨 digit-recognizer, el yazısı ile çizilmiş rakamları tanıyabilen bir Convolutional Neural Network (CNN) projesidir.
TensorFlow ve Keras ile eğitilmiş bir CNN modeli kullanarak yüklediğiniz görseldeki rakamı tahmin eder.

🖼️ Örnek Çıktı

Tahmin Edilen Rakam: 2

🚀 Özellikler
✅ El yazısı rakam (0–9) tahmini
✅ CNN mimarisi ile yüksek doğruluk
✅ MNIST veri kümesi üzerinde eğitilmiş
✅ Gürültü azaltma için thresholding
✅ Python, TensorFlow ve OpenCV tabanlı

🧰 Kullanılan Teknolojiler
Python 3.x

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

🔧 Kurulum
1️⃣ Depoyu klonlayın
bash
Kopyala
Düzenle
git clone https://github.com/kullanici-adiniz/digit-recognizer.git
cd digit-recognizer
2️⃣ Gereksinimleri yükleyin
bash
Kopyala
Düzenle
pip install -r requirements.txt
requirements.txt içeriği:

nginx
Kopyala
Düzenle
tensorflow
numpy
opencv-python
matplotlib
📂 Kullanım
1️⃣ Model dosyasını (model.keras) proje klasörüne yerleştirin.
Model, MNIST veri seti üzerinde eğitilmiş bir CNN’dir.

2️⃣ Tahmin yapmak istediğiniz rakam görselini hazırlayın ve kendimin.png adıyla proje dizinine koyun.
3️⃣ Çalıştırın
bash
Kopyala
Düzenle
python predict.py
🧪 CNN Model Mimarisi
Model MNIST üzerinde eğitilmiş tipik bir CNN’den oluşur:

📐 Conv2D katmanları

🔄 MaxPooling katmanları

🧹 Dropout

🧮 Dense katman (output: 10 sınıf)

Eğitim kodu bu projeye dahil edilmemiştir. MNIST ile yeni bir CNN eğitmek isterseniz TensorFlow’un resmi MNIST CNN örneğini inceleyebilirsiniz:
📄 https://www.tensorflow.org/tutorials/keras/classification
