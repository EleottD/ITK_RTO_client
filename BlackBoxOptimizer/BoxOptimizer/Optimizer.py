"""@package docstring
Исхоныдй класс оптимизатора

More details.
"""
from .BaseOptimizer import BaseOptimizer

import numpy as np
from typing import Callable, Literal, Tuple
import enum


class OptimisationTypes(enum.Enum):
    """
    OptimisationTypes
    ---
    Enum для различных типов задач оптимизации

    """
    minimize = 0
    maximize = 1
    to_target = 2


optimization_transform_funcs = {
    OptimisationTypes.minimize: lambda x, y = None: x,
    OptimisationTypes.maximize: lambda x, y = None: -x,
    OptimisationTypes.to_target: lambda x, y: x - y,
}
"""Словарь преобразования задачи оптимизации для приведения её к виду argmin(f(x))"""



class Optimizer(object):
    """
    Optimizer
    ---
    Функция оптимизации черного ящика
    
    """
    def __init__(self, 
                 optCls : object = None,
                 external_model: Callable[[np.ndarray], np.ndarray] = None,
                 user_function: Callable[[np.ndarray], float] = lambda x: x[0],
                 optimisation_type: OptimisationTypes = OptimisationTypes.minimize,
                 target: float | None = None,
                 *args,
                 **kwargs
                 ) -> None:
        """
        __init__
        ---
        Аргументы:
            optCls : object - выбранный класс оптимизации
            external_model: Callable[[np.ndarray], np.ndarray] - Функция вычисления внешней модели
            user_function: Callable[[np.ndarray], float] = lambda x: x[0] - Пользовательская функция дополнительной обработки выхода внешней модели
            optimisation_type: OptimisationTypes = OptimisationTypes.minimize - Тип задачи оптимизации {минимизация, максимизация, приведение к значению}
            target: float | None = None - Значение, к которому необходимо привести целевую функцию при выборе типа оптимизации to_target
        """

        self._currentOptimizerClass = optCls
        """Текущий оптимизирующий класс"""

        if self._currentOptimizerClass is None:
            raise ValueError("Не указан оптимизируйщий класс")

        self._CurrentOptimizerObject = self._currentOptimizerClass(*args, **kwargs)
        """Текущий объект оптимизации"""

        if 'seed' in kwargs.keys() is not None:
            try:
                self.seed = int(kwargs['seed'])
            except:
                self.seed = None
            """Сид для псевдослучайной генерации"""
            np.random.seed(self.seed)


        self.external_model: Callable[[np.ndarray], np.ndarray] = external_model
        """Текущая внешняя модель - черный ящик"""

        if self.external_model is None:
            raise ValueError("Не указана внешняя модель")

        self.user_function: Callable[[np.ndarray], float] = user_function
        """Пользовательская функция дополнительной обработки выхода внешней модели"""

        self.optimisation_type: OptimisationTypes = optimisation_type
        """Тип задачи оптимизации (минимизация целевой функции, максимизация или приведение к заданному значению)"""

        self.target: float | None = target
        """Значение, к которому необходимо привести целевую функцию при выборе типа оптимизации to_target"""

        self._usage_count: int = 0
        """Число обращений к внешней модели"""



    def configure(self, **kwargs) -> None:
        """
        configure
        ---
        Метод обновления конфигурации
        
        Выполняет конфигурирование параметров работы алгоритмов при наличии необходимых
        
        Пример использования:
        
        >>> import SpecificOptimizer
        ... optimizer = SpecificOptimizer()
        ... optimizer.configure(some_parameter_A = some_value_A, some_parameter_B = some_value_B)
        """

        for key, value in kwargs.items():
            if key == "external_model":
                self._set_model(value)
                """Изменяем параметр в Optimizer"""
            elif key == "_from_model_vec_size":
                self._CurrentOptimizerObject.configure(**{key: value+1})
            elif key in self.__dict__:
                self.__dict__[key] = value
                """Изменяем параметр в Optimizer"""
            else:
                self._CurrentOptimizerObject.configure(**{key: value})
                """Если не нашли параметр в Optimizer, конфигурируем алгоритм оптимизации"""



    def modelOptimize(self) -> None:
        """
        modelOptimize
        ---
        Запуск оптимизации через передачу функции черного ящика
        """
        return self._CurrentOptimizerObject.modelOptimize(self._evaluate)



    def getResult(self) -> np.ndarray:
        """
        getResult
        ---
        Возврат результата работы модуля оптимизации черного ящика
        """
        return self._CurrentOptimizerObject.getResult()



    def getHistoricalData(self, key : None | Literal["vec_to_model", "vec_from_model", "obj_val"] = None) -> None | list:
        """
        getHistoricalData
        ---
        Получение исторических данных, собранных в результате работы алгоритма
        
        "vec_to_model" - Получение истории векторов, отправляемых в оптимизационную модель на протя-
                         жении всех заданных итераций работы
        "vec_from_model" - Получение истории векторов, получаемых от оптимизационной модели на протя-
                           жении всех заданных итераций работы
        "obj_val" - Зафиксированные значения целевой функции
        """
        return self._CurrentOptimizerObject.getHistoricalData(key)
    
    
    
    def setVecItemLimit(self, 
                        index : int, 
                        vec_dir : Literal["to_model", "from_model"] = "to_model",
                        min : None | float = None,
                        max : None | float = None) -> None:
        """
        setVecItemLimit
        ---
        
        Установка ограничения для параметра внутри вектора
        
        Аргументы:
            index   : int - Индекс ограничиваемого элемента внутри вектора
            min     : None | float - Устанавливаемый минимум
            max     : None | float - Устанавливаемый максимум
            vec_dir : Literal["to_model", "from_model"] - Выбор вектора направления передачи параметров
        """
        return self._CurrentOptimizerObject.setVecItemLimit(
            index = index, vec_dir = vec_dir, min = min, max = max)



    def setVecItemType(self, 
                        index : int, 
                        new_type : Literal["float", "bool"],
                        vec_dir  : Literal["to_model", "from_model"] = "to_model",
                        min : None | float = None,
                        max : None | float = None,
                        *args, 
                        **kwargs) -> None:
        """
        setVecItemType
        ---
        
        Установка нового типа для элемента вектора кандидата
        
        Аргументы:
            index    : int - Индекс ограничиваемого элемента внутри вектора
            min      : None | float - Устанавливаемый минимум
            max      : None | float - Устанавливаемый максимум
            vec_dir  : Literal["to_model", "from_model"] - Выбор вектора направления передачи параметров
            new_type : Literal["float", "bool"] - Выбираемый тип для элемента
            *args, **kwargs - Специфичные для типа атрибуты
        """
        return self._CurrentOptimizerObject.setVecItemType(
            index = index, vec_dir = vec_dir, new_type = new_type, *args, **kwargs)



    def setPreSetCadidateVec(self, 
                            candidate_index : int,
                            vec : np.ndarray,
                            vec_dir : Literal["to_model", "from_model"] = "to_model") -> None:
        """
        setPreSetCadidateVec
        ---
        
        Установка значений вектора кандидата по номеру кандидата
        
        Аргументы:
            candidate_index : int - Индекс вектора-кандидата
            vec : np.ndarray - Устанавливаемый вектор
            vec_dir  : Literal["to_model", "from_model"] - Выбор вектора направления передачи параметров
        """
        return self._CurrentOptimizerObject.setPreSetCadidateVec(
            candidate_index = candidate_index,
            vec = vec,
            vec_dir = vec_dir
            )



    def getOptimizer(self):
        """
        getOptimizer
        ---
        Возврат экземпляр модели оптимизации, с которой он работал
        Необходимо для отладки, получения метрик, тестирования
        """
        return self._CurrentOptimizerObject



    def _evaluate_external_model(self, to_vec: np.ndarray) -> np.ndarray:
        """
        evaluate_external_model
        ---
        Метод обращения к внешней модели
        """
        self._usage_count += 1
        return self.external_model(to_vec)
    # TODO: Подумать над возможностью реализовать мемоизацию для снижения числа вызовов внешней модели



    def _apply_user_func(self, from_vec: np.ndarray) -> float:
        """
        apply_user_func
        ---
        Метод применения пользовательской функции к выходу внешней модели
        """

        return self.user_function(from_vec)



    def _transform_optimisation_type(self, value: float) -> float:
        """
        transform_optimisation_type
        ---
        Метод преобразования выхода пользовательской функции для приведения задачи к минимизации
        """

        if self.optimisation_type is OptimisationTypes.to_target and self.target is None:
            raise AttributeError('Не задана цель оптимизации')

        transform_func = optimization_transform_funcs[self.optimisation_type]
        return transform_func(value, self.target)



    def _evaluate(self, to_vec: list[np.ndarray] | np.ndarray) -> list[Tuple[float, np.ndarray]] | Tuple[float, np.ndarray]:
        """
        evaluate
        ---
        Метод вычисления внешней модели для двух вариантов входных данных.
        Вход_v1 - набор кандидатов MV - массив, каждый элемент которого потенциальный кандидат вектор принимаемых внешней моделью MV
        Вход_v2 - np.array вектор с единственным кандидатом
        Выход_v1 - Набор результатов вычисления для каждого кандидата - массив, каждый элемент которого кортеж (целевая переменная, numpy массив возвращаемых CV)
        Выход_v2 - единственный кортеж из целевого значения и numpy массива из остальных CV
        """

        if type(to_vec) is list or type(to_vec) is np.ndarray and len(to_vec.shape) > 1:
            res = []
            for candidate in to_vec:
                # print(candidate)
                from_vec = self._evaluate_external_model(candidate)
                user_val = self._apply_user_func(from_vec)
                optimisation_value = self._transform_optimisation_type(user_val)
                # res.append((optimisation_value, from_vec.copy())) # Версия с кортежем
                res.append(np.array([optimisation_value, *from_vec.copy()])) # Версия с numpy массивом

        elif type(to_vec) is np.ndarray and len(to_vec.shape) == 1:
            from_vec = self._evaluate_external_model(to_vec)
            user_val = self._apply_user_func(from_vec)
            optimisation_value = self._transform_optimisation_type(user_val)
            # res = (optimisation_value, from_vec.copy()) # Версия с кортежем
            res = np.array([optimisation_value, *from_vec.copy()]) # Версия с numpy массивом

        else:
            raise TypeError('Неверно сформированные данные для вычисления во внешней модели')

        return res



    def get_usage_count(self) -> int:
        """
        get_usage_count
        ---
        Возвращает число вызовов внешней модели

        """
        return self._usage_count



    def _refresh_usage_count(self) -> None:
        """
        refresh_usage_count
        ---
        Обнуляет число вызовов внешней модели

        """
        self._usage_count = 0



    def _set_model(self, external_model) -> None:
        """
        set_model
        ---
        Устанавливает объект внешней модели

        """
        self._refresh_usage_count()
        """При смене модели обязательно обнулим число обращений к ней"""
        self.external_model = external_model

