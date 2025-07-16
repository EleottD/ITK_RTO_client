import numpy as np

from typing import Callable, List


from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor


from ..BaseOptimizer import BaseOptimizer, _boolItem

class GaussOpt(BaseOptimizer):
    def __init__(self, kernel_cfg: tuple[str, dict] = ('Matern', {'nu': 2.5}), *args, **kwargs) -> None:
        """
        __init__
        ---
        Конструктор класса оптимизатора
        """
        super().__init__(*args, **kwargs)

        # Динамически подгружаем нужное ядро
        kernel_name, kernel_params = kernel_cfg
        module = __import__('sklearn.gaussian_process.kernels', fromlist=[''])
        kernel_cls = getattr(module, kernel_name)
        kernel = kernel_cls(**kernel_params)

        self.model = GaussianProcessRegressor(kernel=kernel)
        """Модель кригинга, суррогатная модель"""
        self.history_to_opt_model_data : list = []
        """История данных для построения модели"""
        self.res_history_to_opt_model_data : list = []
        """История резльтатов функции для построения модели"""
        self.target_to_opt : bool = False
        """Цель оптимизации False-минимум True-максимум"""
        self.res_of_most_opt_vec : float = 0.0
        """Возрат наилучшего вектора, кандидат на min/max значение функции"""
        self.most_opt_vec = []
        """Кандидат на самый оптимальный вектор"""
        self.input_bound_of_vec = self._bound_func_("to_model")
        """Ограничения параметров векторов на вход в виде массива"""
        self.output_bound_of_vec = self._bound_func_("from_model")
        """Ограничения параметров векторов на выход в виде массива"""
        self.discrete_indices = []
        """Индексы дискретный параметров"""
        self.scaler_x = StandardScaler()
        """Нормализация входных данных"""
        self.scaler_y = StandardScaler()
        """Нормализация выходных данных"""
        self.X_scaled = []
        """Нормированные входные данные"""
        self.Y_scaled = []
        """Нормированные выходные данные"""
        self.bound_x_scaled = []
        """Нормированные входные ограничения"""
        self.bound_y_scaled = []
        """Нормированые выходные ограничения"""
    
    @staticmethod
    def _expected_improvement(x, model, y_opt, maximize):
        x = np.array(x).reshape(1, -1)
        mu, sigma = model.predict(x, return_std=True)   # shapes (1, m), (1, m)
        mu, sigma = mu.flatten(), sigma.flatten()       # теперь 1-D длины m

        if np.isnan(mu[0]) or np.isnan(sigma[0]):
            return 0.0

        # считаем EI только для [0]-го выхода, но храним в массиве
        ei = np.zeros_like(mu)
        imp = (mu[0] - y_opt) if maximize else (y_opt - mu[0])
        if sigma[0] > 0:
            Z = imp / sigma[0]
            ei[0] = imp * norm.cdf(Z) + sigma[0] * norm.pdf(Z)

        # безопасно зануляем EI, если σ[0]==0
        ei[0] = np.where(sigma[0] == 0.0, 0.0, ei[0])

        # возвращаем скаляр
        return -float(ei[0])
    """Функция максимального правдоподобия"""

    def _propose_location(self):
        y_opt    = self.res_of_most_opt_vec
        maximize = self.target_to_opt
        # базовая функция (EI)
        func = lambda x: self._expected_improvement(x, self.model, y_opt, maximize)

        # создаём список ограничений на каждый выход beyond bounds
        constraints = []
        # model.predict возвращает [f, g1, g2, ...], где g — вторые выходы
        for i, (lb, ub) in enumerate(self.bound_y_scaled):
            # g_i(x) >= lb  ⇒  g_i(x) - lb ≥ 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, lb=lb:  self.model.predict(x.reshape(1, -1))[0, i] - lb
            })
            # g_i(x) ≤ ub  ⇒  ub - g_i(x) ≥ 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, ub=ub: ub - self.model.predict(x.reshape(1, -1))[0, i]
            })

        res = minimize(
            fun=func,
            x0 = self.most_opt_vec,
            bounds=self.bound_x_scaled,
            constraints=constraints,
            method='SLSQP',
            options={'ftol':1e-6, 'maxiter':200}
        )
        return res.x
    """Функция высчитывающая следущую точку для подсчета"""

    def _bound_func_(self, vec_dir: str):
        bounds = np.array([])
        info = np.iinfo(np.int64)
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        # Безопасный доступ к границам через properties_list
        for i in range(size):
            if i < len(vec_data._values_properties_list):
                lower[i] = vec_data._values_properties_list[i].min
                upper[i] = vec_data._values_properties_list[i].max
    
        for min_v, max_v in zip(lower, upper):
            bounds = np.append(bounds, (min_v, max_v), axis=0)

        bounds = bounds.reshape(size, 2) 
        for vec in bounds:
            vec[0] = vec[0] if not np.isinf(vec[0]) else info.min
            vec[1] = vec[1] if not np.isinf(vec[1]) else info.max
        return bounds
    """Функция возращяющая массив минимумов и максимумов вида [[min1, max1],[min2, max2]...]"""

    def _init_vecs(self,population):
            # if self._seed is not None:
            #     np.random.seed(self._seed)
            # Преобразуем входные данные в numpy array для удобства
            constraints = np.asarray(self.input_bound_of_vec)
            # Генерируем данные для каждой компоненты отдельно
            components = [
                np.random.uniform(low=min_max[0], high=min_max[1], size=population*self._to_model_vec_size)
                for min_max in constraints
            ]
            
            # Транспонируем результат, чтобы векторы были строками матрицы
            return np.column_stack(components)
    """Создание первой популяции векторов для нормальной работы метода, необходимо 10*количество MV"""

    def _add_point(self):
        # Преобразуем входные данные в numpy array для удобства
        constraints = np.asarray(self.input_bound_of_vec)
        # Генерируем данные для каждой компоненты отдельно
        components = [
                np.random.uniform(low=min_max[0], high=min_max[1], size=1)
                for min_max in constraints
        ]  
        # Транспонируем результат, чтобы векторы были строками матрицы
        return np.column_stack(components)
    
    def _main_calc_func(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model.fit(self.X_scaled,self.Y_scaled)
        next_x = self._propose_location()
        self.X_scaled = np.vstack([self.X_scaled, next_x.copy()])

        next_x_for_fun = self.scaler_x.inverse_transform(next_x.reshape(1, -1))
        self.history_to_opt_model_data = np.vstack([self.history_to_opt_model_data, *next_x_for_fun])
        for idx in self.discrete_indices:
            next_x_for_fun[0][idx] = 1 if next_x_for_fun[0][idx]>=0.5 else 0
        output_value = func(next_x_for_fun[0])
        scaled_output_value = self.scaler_y.transform([output_value])
        self.res_history_to_opt_model_data.append(output_value)
        self.Y_scaled = np.vstack([self.Y_scaled, scaled_output_value])
        output_value_scaled = self.scaler_y.transform([output_value])[0]
        if self._check_output_constraints(output_values=output_value):
            if self.target_to_opt and output_value_scaled[0]>self.res_of_most_opt_vec:
                self.res_of_most_opt_vec = output_value_scaled[0]
                self.most_opt_vec = next_x
                
            if not self.target_to_opt and output_value_scaled[0]<self.res_of_most_opt_vec:
                self.res_of_most_opt_vec = output_value_scaled[0]
                self.most_opt_vec = next_x
    """Основная функция подсчета"""

    def _check_output_constraints(self, output_values) -> bool:
        """Проверка ограничений выходных переменных"""
        if len(output_values) <= 1:  # Только целевая функция
            return True
            
        # Проверяем только те параметры, для которых заданы ограничения
        num_output_params = min(len(self.output_bound_of_vec), len(output_values)-1)
        
        for i in range(num_output_params):
            if (output_values[i+1] < self.output_bound_of_vec[i][0] or 
                output_values[i+1] > self.output_bound_of_vec[i][1]):
                return False
        
        return True

    def configure(self, **kwargs):
        kernel_cfg = kwargs.pop('kernel_cfg', None)

        super().configure(**kwargs)

        if kernel_cfg is not None:
            kernel_name, kernel_params = kernel_cfg

            module = __import__('sklearn.gaussian_process.kernels', fromlist=[''])
            kernel_cls = getattr(module, kernel_name)

            kernel = kernel_cls(**kernel_params)

            self.model = GaussianProcessRegressor(kernel=kernel)
        if 'discrete_indices' in kwargs:
            self.discrete_indices = kwargs['discrete_indices']
    """Настройка метода"""

    def modelOptimize(self, func : Callable[[np.array], np.array]) -> None:
        self.input_bound_of_vec = self._bound_func_("to_model")
        self.output_bound_of_vec = self._bound_func_("from_model")

        discrete_indices = []
        for i, prop in enumerate(self._to_opt_model_data._values_properties_list):
            if isinstance(prop, _boolItem):
                discrete_indices.append(i)
        
        # Добавляем явно указанные дискретные индексы
        discrete_indices.extend(self.discrete_indices)
        self.discrete_indices = list(set(discrete_indices))  # Удаляем дубликаты

    
        self.history_to_opt_model_data = self._init_vecs(20)
        
        history_for_fun = self.history_to_opt_model_data
        for vec in history_for_fun:
            for idx in self.discrete_indices:
                vec[idx] = 1 if vec[idx]>=0.5 else 0

        res_list = []

        for vec in history_for_fun:
            output_value = func(vec)
            res_list.append(output_value)

        self.res_history_to_opt_model_data = res_list

        self.scaler_x.fit(self.history_to_opt_model_data)
        self.scaler_y.fit(self.res_history_to_opt_model_data)

        self.X_scaled = self.scaler_x.transform(self.history_to_opt_model_data)
        self.Y_scaled = self.scaler_y.transform(self.res_history_to_opt_model_data)

        means = self.scaler_x.mean_
        scales = self.scaler_x.scale_
        self.bound_x_scaled = [((lb - means[i]) / scales[i], (ub - means[i]) / scales[i])
                                    for i, (lb, ub) in enumerate(self.input_bound_of_vec)]

        ###
        
        self.res_of_most_opt_vec = np.iinfo(np.int64).min if self.target_to_opt else np.iinfo(np.int64).max
        for vec in np.array(self.Y_scaled):
            if self._check_output_constraints(output_values=self.scaler_y.inverse_transform([vec])[0]):
                if self.target_to_opt:
                    self.res_of_most_opt_vec=vec[0] if vec[0]>self.res_of_most_opt_vec else self.res_of_most_opt_vec
                    
                else:
                    self.res_of_most_opt_vec=vec[0] if vec[0]<self.res_of_most_opt_vec else self.res_of_most_opt_vec


        if self.res_of_most_opt_vec == np.iinfo(np.int64).min or self.res_of_most_opt_vec==np.iinfo(np.int64).max:
            while True:
                point = self._add_point()
            
                self.history_to_opt_model_data = np.vstack((self.history_to_opt_model_data, point[0]))
                point_for_fun=point[0]
                for idx in self.discrete_indices:
                    point_for_fun[idx] = 1 if vec[idx]>=0.5 else 0
                y_point = func(point_for_fun)
                self.res_history_to_opt_model_data.append(y_point)
                if self._check_output_constraints(output_values=y_point):
                    break
            self.Y_scaled = self.scaler_y.transform(self.res_history_to_opt_model_data)
            self.X_scaled = self.scaler_x.transform(self.history_to_opt_model_data)
            self.res_of_most_opt_vec = self.Y_scaled[-1,0]


        self.most_opt_vec = self.X_scaled[np.array(self.Y_scaled)[:,0].tolist().index(self.res_of_most_opt_vec)]

        ###
        for _ in range(self._iteration_limitation):
            self.input_bound_of_vec = self._bound_func_("to_model")
            means_x = self.scaler_x.mean_
            scales_x = self.scaler_x.scale_
            self.bound_x_scaled = [((lb - means_x[i]) / scales_x[i], (ub - means_x[i]) / scales_x[i])
                                        for i, (lb, ub) in enumerate(self.input_bound_of_vec)]
            
            self.output_bound_of_vec = self._bound_func_("from_model")
            means_y = self.scaler_y.mean_
            scales_y = self.scaler_y.scale_
            self.bound_y_scaled = [((lb - means_y[i]) / scales_y[i], (ub - means_y[i]) / scales_y[i])
                                        for i, (lb, ub) in enumerate(self.output_bound_of_vec)]
            
            self._main_calc_func(func=func)
            
    """Функция инициализации и оптимизации"""

    def getResult(self) -> np.array:
        result = self.most_opt_vec
        true_result = self.scaler_x.inverse_transform([result])
        for idx in self.discrete_indices:
            true_result[0,idx] = 1 if result[idx]>=0.5 else 0
        return true_result[0]
    """Функция результата, возращает точку"""

    def get_y(self) -> List:
        result = self.most_opt_vec
        i = np.where((self.history_to_opt_model_data == result).all(axis=1))[0][0]
        return self.res_history_to_opt_model_data[i]