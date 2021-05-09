import tensorflow as tf
from tensorflow import keras as ks

# Подключения, встроенного набора данных, для обучения НС
# (train_i, train_l) - набор данных для обучения НС
# (test_i, test_l) - тестовый набор данных для проверки НС
# i(image) - само изображение, l(labels) - метка у изображения, указывающая на класс, к которому оно относится
fm = ks.datasets.fashion_mnist
(train_i, train_l), (test_i, test_l) = fm.load_data()


# Создание НС для распознавания обуви
model = ks.Sequential()
# Данная НС содержет в себе 3 слоя
# Первый слой принимает изображение размером 28x28
model.add(ks.layers.Flatten(input_shape=(28, 28)))
# Второй слой содержет в себе функции, которые обрабатывают изображение по параметрам
# relu - функция активации, которая просто возвращает значение, если оно больше 0 и отсеивает остальные
model.add(ks.layers.Dense(128, activation=tf.nn.relu))
# Третий слой выдает одно из 10 значений
# softmax - функция мягкого максимума, выдает значение максимального нейрона
model.add(ks.layers.Dense(10, activation=tf.nn.softmax))


# Передача в Нс функции потери и функции оптимизации
model.compile(optimizer=ks.optimizers.Adam(), loss='sparse_categorical_crossentropy')


# Обучение НС
model.fit(train_i, train_l, epochs=20)

classifications = model.predict(test_i)


print(classifications[1])
print(test_l[1])
