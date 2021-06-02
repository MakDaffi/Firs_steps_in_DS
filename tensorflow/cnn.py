from tensorflow import keras as ks

# Код для реализации сврточной нейронной сети(теоретическая реализация без обучения)
# Создание НС
model = ks.Sequential()
# Добавление к НС сверточных слоев нейронов
model.add(ks.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(ks.layers.MaxPooling2D(2, 2))
model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(ks.layers.MaxPooling2D(2, 2))
# Добавление к НС слоев нейронов
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(128, activation='relu'))
model.add(ks.layers.Dense(10, activation='softmax'))
