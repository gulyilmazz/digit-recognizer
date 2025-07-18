import tensorflow as tf
import matplotlib.pyplot as plt

#MNIST 0-9 arası el yazısı rakamlar.

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1) #sebebini CNN'e geçince konuşacağız.

plt.figure(figsize=(10, 10))

for i in range(10):
    plt.subplot(10, 10, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.suptitle("MNIST Veri Seti ilk 10 görsel")
plt.show()

model = tf.keras.models.Sequential([
    # kernel_size: filtre boyutu
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'), #128 nöronlu katman => karar mekanizması
    tf.keras.layers.Dense(10, activation='softmax') #10 nöronlu katman => çıktı katmanı (0-9 arası rakamlar)
    #softmax: 0-9 arası rakamların olasılıklarını hesaplar.
    #softmax => 10 tahmin yapar, en yüksek değere sahip olan tahmin doğru tahmindir.
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test başarımı: {test_acc}")

model.save("model.keras")