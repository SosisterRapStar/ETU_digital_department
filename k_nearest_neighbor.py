import numpy as np



class KNearestNeighbor:
    """ KNN классификатор для меры Евклида (L2 меры) """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Конструкция fit просто запоминает design matrix (выборку) и целевые переменные
        тренировочной части данных
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Функция для предсказания значений вызывает разные реализации алгоритма
        2ой цикл
        1 цикл
        Векторизованная реализация без циклов
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Реализация с двумя циклами
        """
        num_test = X.shape[0] # кол-во тестовых объектов - строк матрицы
        num_train = self.X_train.shape[0] # количество тренировочных объектов
        dists = np.zeros((num_test, num_train)) # матрица с расстояниями между каждым тренировочным объектом и тестовым
        for i in range(num_test):
            for j in range(num_train):

                diffs = X[i] - self.X_train[j] # вектор где каждое значение - разность тестового вектора и тренировочного
                diffs = np.square(diffs) # возведение в квадрат
                diffs = np.sum(diffs) # суммирование
                diffs = np.sqrt(diffs) # взятие корня из суммы
                dists[i, j] = diffs # кладем расстояние в ячейку расстояний

        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            dists_row = np.square(self.X_train - X[i]) # вектор с разностями координат в квадрате, только в этом случае используем
            # broadcasting в numpy и сразу 1 вектор вычитаем из всех векторов в матрице
            dists_row = np.sum(dists_row, axis=1) # суммируем по оси строк
            dists_row = np.sqrt(dists_row)  # берем корень
            dists[i] = dists_row # кладем в строку матрицы

        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # если разложить расстояние между векторами
        # получим что dist(x = [a, b, c]* y = [i, j, k]) это
        # (a - i)^2 + (b - i)^2 + (c-k)^2 забьем на корень
        # это тогда a^2 + b^2 + c^2 + i^2 + j^2 + k^2 - 2(ai + bj + ck)(скалярное умножение <x, y>)
        I = np.sum(np.square(self.X_train), axis=1) # матрица где суммы квадратов координат тренинг объектов
        B = np.sum(np.square(X), axis=1) # матрица где суммы квадратов координат тест объектов
        middle_matrix = I.reshape(len(I), 1) + B # так как summ собирает все в одномерный массив, то нельзя просто
        # применить I.T - он просто упакует его в двумерный, нужно сделать reshape чтобы broadcast add это смог сложить
        l2_distances = np.dot(self.X_train, X.T) # считает скалярное умножение для того чтобы потом умножить на -2
        dists = middle_matrix - 2 * l2_distances # получили матрицу
        dists = np.sqrt(dists.T)# матрицу еще перевернуть нужно

        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            k_nearest_idexes = dists[i].argsort()[:k] # argsort возвращает индексы в порядке возрастания элементов под этими индексами
            closest_y = self.y_train[k_nearest_idexes] # возвращает самые близкие лейблы
            # возвращает самые частые близкие лейблы
            y_pred[i] = np.bincount(closest_y).argmax() # вообще если бы значение могли быть отрицательны
            # то bincount должен сломаться
        return y_pred