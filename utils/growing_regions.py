from ast import arg
import threading
from common import ConnectedLimiar, ConnectedPixel, MovingMeanLimiar
from common import determina_lista_pesquisa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils

def bfs_factory(img_: np.ndarray, thresh: int, color: tuple, neighboor_size: tuple = (3,3)):
    cp = img_.copy()
    pesquisados = np.zeros(cp.shape)
    def bfs(start_point: ConnectedPixel,
            img: np.ndarray = cp, 
            pesquisados: np.ndarray = pesquisados, 
            neighboor_size: tuple = neighboor_size,
            thresh: int = thresh, 
            color: int = color, i_=0):
        queue: list[ConnectedPixel] = [start_point]
        while True:
            try:
                (i, j) = queue[0].point_
            except IndexError:
                return img
            if pesquisados.sum() == pesquisados.shape[0] * pesquisados.shape[1]:
                return img
            eh_continuo = queue[0].determina_continuidade_no_root(img_, thresh)
            if pesquisados[i, j] :
                queue.pop(0)
                i_ += 1
                continue
            pesquisados[i][j] = 1
            if not eh_continuo:
                queue.pop(0)
                i_ += 1
                continue
            img[i,j] = color
            for point, root in determina_lista_pesquisa(pesquisados, i, j, neighboor_size[1], neighboor_size[0]):
                if not pesquisados[point[0], point[1]]:
                    queue.append(queue[0].cria_ramo(point, img_))
            queue.pop(0)
            if len(queue) != 0:
                i_ += 1
                continue
            return img

    def get():
        return cp

    return bfs, get

img = cv.imread('images/test.jpg', 0)
img_ = imutils.resize(img, width=200)
bfs, get = bfs_factory(img_, 40, 255, (2, 2))

start_point = MovingMeanLimiar(point=(104, 148), current_mean=img_[2*52, 2*74])
cp = bfs(start_point)
_, cp = cv.threshold(cp, 254, 255, cv.THRESH_BINARY)
# plt.imshow(bin, cmap='gray')
plt.imshow(cp, cmap='gray')
plt.show()