import os
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import sklearn
import scipy
from scipy import misc
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import square
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from antigate import AntiGate
from python3_anticaptcha import ImageToTextTask
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers


from keys import anticaptcha_key

DEBUG = True

def get_images_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name)


def save_nparray_as_imagem(array, caminho):

    pil_img = Image.fromarray(array.astype(np.uint8)*255)
    pil_img.save(caminho)


def preprocessamento(imagempath):
    # mode L = (8-bit pixels, black and white)
    imagem = np.array(Image.open(imagempath).convert('L'))

    arr = imagem < 10
    if DEBUG:
        save_nparray_as_imagem(arr, "./data/debug/preto_e_branco.png")

    eroded = morphology.erosion(arr, np.array(
        [[1],
        [1]]
    ))
    if DEBUG:
        save_nparray_as_imagem(eroded, "./data/debug/eroded.png")

    sem_pontos = morphology.remove_small_objects(eroded, min_size=8)
    if DEBUG:
        save_nparray_as_imagem(sem_pontos, "./data/debug/remove_pequenos_objetos.png")

    return sem_pontos


def split(arrayimage):
    partes = []
    for i in range(6):
        parte = arrayimage[:, 12 + (28 * i):12 + (28 * (i + 1))]

        soma_vertical = np.sum(parte, axis=1)
        soma_grupos = np.zeros(soma_vertical.shape[0]-28)
        for aux in range(soma_vertical.shape[0]-28):
            soma_grupos[aux] = np.sum(soma_vertical[aux: aux + 28])
        inicio = np.argmax(soma_grupos)

        partes.append(parte[inicio:inicio+28, :])
        if DEBUG:
            save_nparray_as_imagem(parte[inicio:inicio+28, :], f"./data/debug/parte{i}.png")

    return partes


def split_all(pasta_origem, pasta_destino):
    paths = get_images_files(pasta_origem)
    for filename in paths:
        nome = Path(filename).stem
        preprocessada = preprocessamento(filename)
        chars = split(preprocessada)
        for i, c in enumerate(chars):
            # nome_pasta = nome[i] if nome[i].islower() else ("$" + nome[i])
            nome_pasta = nome[i]
            os.makedirs(pasta_destino / f"{nome_pasta}", exist_ok=True)
            save_nparray_as_imagem(c, pasta_destino / f"{nome_pasta}" / (str(uuid.uuid4()) + '.png'))


def solve_all_antigate(pasta_origem, pasta_destino):

    def decode(path):
        anticaptcha = ImageToTextTask.ImageToTextTask(anticaptcha_key=anticaptcha_key)
        resposta1 = anticaptcha.captcha_handler(captcha_file=path)['solution'].get('text', None)
        resposta2 = anticaptcha.captcha_handler(captcha_file=path)['solution'].get('text', None)
        if resposta1 is not None and resposta1 == resposta2:
            os.rename(path, pasta_destino / f"{resposta1}.png")

    paths = get_images_files(pasta_origem)
    with ThreadPoolExecutor(max_workers=30) as executor:
        for path in paths:
            executor.submit(decode, path)


def gera_svm_char_model(path):
    paths = list(get_images_files(path))
    Y = np.array([ Path(path).parent.name for path in paths])
    X = np.array([ np.array(Image.open(path).convert('1')).flatten() for path in paths])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

    clf = SVC(kernel="linear", C=0.025)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

    print(len([1 for i, j in zip(clf.predict(X_test), y_test) if i.replace("$", "").lower() == j.replace("$", "").lower()]) / len(y_test))
    print( [f"{i} - {j}" for i, j in zip(clf.predict(X_test), y_test) if i != j])


def gera_tf_char_model(path):
    gerador = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = gerador.flow_from_directory(
        path,
        target_size=(28, 28),
        batch_size=442,
        class_mode='categorical',
        color_mode='grayscale')

    modelo = get_modelo((28,28,1), 35)
    data = train_generator.next()
    modelo.fit_generator(train_generator,
              epochs=10,
    )

def gera_tf_char_model2(path):

    class_to_int = lambda x: '123456789abcdefghijklmnopqrstuvwxyz'.find(x.lower())

    paths = list(get_images_files(path))
    Y = np.array([class_to_int(Path(path).parent.name) for path in paths])
    X = np.array([np.array(Image.open(path).convert('1')).reshape((28,28,1)).astype(np.float32) for path in paths])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    y_train = tf.keras.utils.to_categorical(y_train, 35)
    y_test = tf.keras.utils.to_categorical(y_test, 35)

    modelo = get_modelo((28,28,1), 35)

    modelo.fit(X_train, y_train,
              batch_size=32,
              epochs=50,
              verbose=1,
              validation_data=(X_test, y_test)
               )
    score = modelo.evaluate(X_test, y_test, verbose=0)
    print(score)


def get_modelo(imput_shape, out_shape):
    model = tf.keras.Sequential()
    model.add(layers.Convolution2D(32, (5, 5), activation='relu', input_shape=imput_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Convolution2D(128, (5, 5), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    # print(model.output_shape)
    model.add(layers.Dense(out_shape, activation='softmax'))
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy', "accuracy"])

    return model


if __name__ == '__main__':
    split_all(Path("./data/classificadas"), Path("./data/chars"))
    # solve_all_antigate(Path("./data/originais"), Path("./data/classificadas"))
    # gera_svm_char_model(Path("./data/chars"))
    # gera_tf_char_model2(Path("./data/chars"))
    pass