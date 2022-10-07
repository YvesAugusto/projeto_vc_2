from common import ConnectedLimiar, ConnectedPixel, MovingMeanLimiar
from common import determina_lista_pesquisa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils

def bfs_factory(img_: np.ndarray, start_point: ConnectedPixel, thresh: int, color: tuple, neighboor_size: tuple = (3,3)):
    cp = img_.copy()
    def bfs(img: np.ndarray = cp, queue: list[ConnectedPixel] = [start_point],
            pesquisados: np.ndarray = np.zeros(cp.shape), neighboor_size: tuple = neighboor_size,
            thresh: int = thresh, color: int = color, cls: ConnectedPixel = type(start_point), i_=0):
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
    return bfs

img = cv.imread('../images/test.jpg', 0)
img_ = imutils.resize(img, width=100)
start_point = MovingMeanLimiar(point=(40, 60), current_mean=img_[40, 60])
# start_point = ConnectedLimiar(point=(40, 60))
bfs = bfs_factory(img_, start_point, 51, 255, (2,2))
cp = bfs()
plt.imshow(cp)
plt.show()