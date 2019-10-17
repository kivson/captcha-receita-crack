import os
import string
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

CARACTERS = string.digits + string.ascii_lowercase
QUANTIDADE_DE_LETRAS = 6

def get_images_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name)


def class_array_to_string(array):
    caracters = CARACTERS
    array = np.reshape(array, (int(array.shape[0] / len(caracters)), len(caracters)))
    return ''.join([caracters[i] for i in array.argmax(axis=1)])


def string_to_class_array(respostas):
    caracters = CARACTERS
    mapa_char_int = {caracters[i]: i for i in range(len(caracters))}

    tamanhos_resposta = len(respostas[0])
    numero_caracters = len(caracters)

    resp = []

    for resposta in respostas:
        assert len(resposta) == tamanhos_resposta
        arr = np.zeros((tamanhos_resposta * numero_caracters))
        for i, c in enumerate(resposta):
            if c not in mapa_char_int:
                # os.remove('resolvidos/{0}.jpg'.format(resposta))
                # break
                raise Exception("Erro ao processar " + resposta + " caracter nao consta na lista de caracteres válidos")
            arr[i * numero_caracters + mapa_char_int[c]] = 1
        resp.append(arr)
        assert class_array_to_string(arr) == resposta

    return np.array(resp)


def train_model(path):

    paths = list(get_images_files(path))

    Y = np.array([Path(path).stem.lower() for path in paths])
    X = np.array([np.array(Image.open(path).convert('RGB')).astype(np.float32) for path in paths])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    y_train = string_to_class_array(y_train)
    y_test = string_to_class_array(y_test)

    modelo = get_model((50, 180, 3), QUANTIDADE_DE_LETRAS * len(CARACTERS))

    modelo.fit(X_train, y_train,
              batch_size=32,
              epochs=200,
              verbose=1,
              validation_data=(X_test, y_test)
               )
    score = modelo.evaluate(X_test, y_test, verbose=1)

    previsoes = modelo.predict(X_test)
    prev_string = list(map(class_array_to_string, previsoes))
    resp_string = list(map(class_array_to_string, y_test))
    avaliacao = len(list(filter(lambda x: x[0] == x[1], zip(prev_string, resp_string)))) * 1. / len(prev_string)
    print("Avaliacao teste", avaliacao)

    print(score)

def get_model(imput_shape, out_shape):
    model = tf.keras.Sequential()
    model.add(layers.Convolution2D(64, (5, 5), activation='relu', input_shape=imput_shape))
    # model.add(layers.Convolution2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Convolution2D(128, (5, 5), activation='relu'))
    # model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Convolution2D(256, (5, 5), activation='relu'))
    # model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    # print(model.output_shape)
    model.add(layers.Dense(out_shape, activation='sigmoid'))
    # 8. Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["categorical_accuracy", "binary_accuracy"])

    return model

if __name__ == '__main__':
    train_model(Path("./data/classificadas"))
    pass