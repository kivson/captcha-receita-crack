from pathlib import Path
import numpy as np
from PIL import Image
import sklearn
import scipy
from scipy import misc
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import square


def save_nparray_as_imagem(array, caminho):

    pil_img = Image.fromarray(array.astype(np.uint8)*255)
    pil_img.save(caminho)


def preprocessamento(imagempath):
    # mode L = (8-bit pixels, black and white)
    imagem = np.array(Image.open(imagempath).convert('L'))

    arr = imagem < 10
    save_nparray_as_imagem(arr, "./data/debug/preto_e_branco.png")

    eroded = morphology.erosion(arr, np.array(
        [[1],
        [1]]
    ))
    save_nparray_as_imagem(eroded, "./data/debug/eroded.png")

    sem_pontos = morphology.remove_small_objects(eroded, min_size=8)
    save_nparray_as_imagem(sem_pontos, "./data/debug/remove_pequenos_objetos.png")

    return sem_pontos


def split(arrayimage):
    partes = []
    for i in range(6):
        parte = arrayimage[:, 12 + (28 * i):12 + (28 * (i + 1))]
        partes.append(parte)
        save_nparray_as_imagem(parte, f"./data/debug/parte{i}.png")


if __name__ == '__main__':
    preprocessada = preprocessamento(Path("./data/originais/0e394dec-7d52-4122-8fa2-fbfb3826d807.png"))
    split(preprocessada)