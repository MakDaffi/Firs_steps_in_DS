from tensorflow import keras as ks

# Создание модели нейронной сети (Sequential)
# Создание слоя ks.layers.Dense
# units - количество нейронов
# input_shape подача значений в нейронную сеть
model = ks.Sequential([ks.layers.Dense(units=1, input_shape=[1])])
# Передача в НС функции потери и функции оптимизации
model.compile(optimizer='sgd', loss='mean_squared_error')


# Передача начальных параметров
xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]


# Обучение НС
model.fit(xs, ys, epochs=500)


# Вывод предсказания значения y при некотором x
print(model.predict([100.0]))