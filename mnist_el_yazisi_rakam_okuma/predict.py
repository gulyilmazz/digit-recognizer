import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Görsel yolu
image_path = "kendimin.png"
model_path = "model.keras"

# Görseli gri tonlamalı okuma yaptım
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")


original_img = cv2.resize(img, (280, 280))


img_resized = cv2.resize(img, (28, 28))

# Gürültüyü azaltmak için threshold uyguladım ***
_, img_thresh = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY_INV)

# Normalizasyon (0-1 aralığına)
img_norm = img_thresh.astype("float32") / 255.0


img_input = img_norm.reshape(1, 28, 28, 1)


model = tf.keras.models.load_model(model_path)


predictions = model.predict(img_input)
predicted_label = np.argmax(predictions)


plt.figure(figsize=(4, 4))
plt.imshow(original_img, cmap="gray")
plt.title(f"Tahmin Edilen Rakam: {predicted_label}", fontsize=16)
plt.axis("off")
plt.show()

print(f"Modelin tahmini: {predicted_label}")
