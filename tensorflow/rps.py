from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras as ks


# Подготовка данных для обучения НС

TRAINING_DIR = "D:/pythonProject/Data Science/rps/"
training_datagen = ImageDataGenerator(rescale=1./255)


train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(150, 150), class_mode='categorical')


VALIDATION_DIR = "D:/pythonProject/Data Science/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              class_mode='categorical')


model = ks.Sequential()
model.add(ks.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(ks.layers.MaxPooling2D(2, 2))
for i in range(3):
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.MaxPooling2D(2, 2))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(512, activation='relu'))
model.add(ks.layers.Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                    validation_steps=3)

