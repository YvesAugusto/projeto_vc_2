import abc
class ConnectedPixel:

    def __init__(self) -> None:
        pass

    @property
    def storage(self):
        pass

    @abc.abstractmethod
    def determina_continuidade_no_root(self):
        pass

    @abc.abstractmethod
    def cria_ramo(self):
        pass

class ConnectedLimiar(ConnectedPixel):

    def __init__(self, point, root = None) -> None:
        self.point = point
        self.root = root
        if root is None:
            self.root = point

    @property
    def point_(self):
        return self.point

    def determina_continuidade_no_root(self, img_, thresh):
        (i, j), (p_i, p_j) = self.point, self.root
        if abs(int(img_[i, j]) - int(img_[p_i, p_j])) < thresh:
            return True
        return False

    def cria_ramo(self, dst, img_):
        return ConnectedLimiar(point=dst, root=self.point)
        

    def __repr__(self):
        return f'point: {self.point}, origin: {self.root}'

class WatershedLimiar(ConnectedPixel):

    def __init__(self, point, root = None) -> None:
        self.point = point
        self.root = root
        if root is None:
            self.root = point

    @property
    def point_(self):
        return self.point

    def determina_continuidade_no_root(self, img_, thresh):
        (i, j), (p_i, p_j) = self.point, self.root
        if img_[i, j] > img_[p_i, p_j]:
            return True
        return False

    def cria_ramo(self, dst, img_):
        return ConnectedLimiar(point=dst, root=self.point)
        

    def __repr__(self):
        return f'point: {self.point}, origin: {self.root}'

class MovingMeanLimiar(ConnectedPixel):

    def __init__(self, point, current_mean, count=None) -> None:
        self.point = point
        self.current_mean = current_mean
        self.count = count
        if count is None:
            self.count = 1

    @property
    def point_(self):
        return self.point

    def determina_continuidade_no_root(self, img_, thresh):
        (i, j) = self.point
        if abs(int(img_[i, j]) - self.current_mean) < thresh:
            return True
        return False

    def cria_ramo(self, dst, img):
        next_mean = self.current_mean + ((int(img[dst[0], dst[1]]) - self.current_mean)/(self.count + 1))
        return MovingMeanLimiar(point=dst, current_mean=next_mean, count=self.count + 1)
        

    def __repr__(self):
        return f'point: {self.point}, origin: {self.root}'





def determina_lista_pesquisa(pesquisados, i, j, nx, ny):
    y, x = pesquisados.shape[0], pesquisados.shape[1]
    lista = []
    for p in range(i-ny, i+ny+1):
        for q in range(j-nx, j+ny+1):
            if p == i and q == j:
                continue
            point = (max(min(y-1, p), 0), max(min(x-1, q), 0))
            lista.append([point, (i, j)])
    return lista