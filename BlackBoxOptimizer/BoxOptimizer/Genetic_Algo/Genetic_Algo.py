from typing import Callable, Optional, List
import numpy as np
from ..BaseOptimizer import BaseOptimizer
from .OptClass import GeneticAlgorithmOptimizer

class Genetic_Algo(BaseOptimizer):
    def __init__(self, 
                to_model_vec_size: int,
                from_model_vec_size: int,
                iter_limit: int,
                seed: int = None,
                population_size: int = 100,
                init_mutation: float = 0.1,
                min_mutation: float = 0.001,
                elite_size: int = 20,
                discrete_indices = None,
                **kwargs) -> None:
        super().__init__(
            to_model_vec_size=to_model_vec_size,
            from_model_vec_size=from_model_vec_size,
            iter_limit=iter_limit,
            seed=seed
        )
        self.seed = seed
        self.dimension = to_model_vec_size
        self.population_size = population_size
        self.generations = iter_limit
        self.init_mutation = init_mutation
        self.min_mutation = min_mutation
        self.elite_size = elite_size
        self.discrete_indices = discrete_indices if discrete_indices is not None else []
        self.best_output_values = None
        self.optimization_history = None  # Добавлено для хранения истории

    def configure(self, **kwargs):
        super().configure(**kwargs)
        for key in ['dimension', 'output_lower_bounds', 'output_upper_bounds', 'discrete_indices']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def _get_bounds(self, vec_dir: str):
        """Получаем ограничения для указанного направления"""
        vec_data = self._to_opt_model_data if vec_dir == "to_model" else self._from_model_data
        size = self._to_model_vec_size if vec_dir == "to_model" else self._from_model_vec_size
        
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        
        for i in range(size):
            if i < len(vec_data._values_properties_list):
                lower[i] = vec_data._values_properties_list[i].min
                upper[i] = vec_data._values_properties_list[i].max
            
        return lower, upper

    def modelOptimize(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        input_lower, input_upper = self._get_bounds("to_model")
        output_lower, output_upper = self._get_bounds("from_model")
        
        # Создаем экземпляр оптимизатора
        ga = GeneticAlgorithmOptimizer(
            func=func,
            dimension=self._to_model_vec_size,
            population_size=self.population_size,
            generations=self.generations,
            init_mutation=self.init_mutation,
            min_mutation=self.min_mutation,
            elite_size=self.elite_size,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
            output_lower_bounds=output_lower,
            output_upper_bounds=output_upper,
            discrete_indices=self.discrete_indices
        )
        
        # Запускаем оптимизацию и собираем историю
        best_x, best_f, best_outputs, *_ = ga.run()
        
        # Сохраняем историю из атрибутов GA
        self.optimization_history = [
            {
                'generation': gen,
                'best_solutions': ga.best_individuals_history[gen],
                'best_fitness': ga.best_fitness_history[gen],
                'average_fitness': ga.avg_fitness_history[gen],
                'mutation_rate': ga.mutation_rates[gen],
                'valid_solutions': ga.num_valid_solutions_history[gen]
            }
            for gen in range(self.generations)
        ]
        
        # Сохраняем результаты
        for to_vec in self._to_opt_model_data.iterVectors():
            to_vec[:] = best_x
        self.best_output_values = best_outputs

    def get_optimization_history(self) -> List[dict]:
        """Возвращает историю оптимизации в виде списка словарей"""
        return self.optimization_history

    def get_history_as_dataframe(self):
        """Возвращает историю в виде DataFrame (требует pandas)"""
        import pandas as pd
        return pd.DataFrame(self.optimization_history)

    def plot_optimization_history(self):
        """Визуализирует историю (требует matplotlib)"""
        if not self.optimization_history:
            raise ValueError("История оптимизации пуста. Сначала запустите modelOptimize.")
            
        import matplotlib.pyplot as plt
        generations = [entry['generation'] for entry in self.optimization_history]
        best_fitness = [entry['best_fitness'] for entry in self.optimization_history]
        avg_fitness = [entry['average_fitness'] for entry in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Лучшее значение')
        plt.plot(generations, avg_fitness, 'g--', label='Среднее значение')
        plt.xlabel('Поколение')
        plt.ylabel('Значение функции')
        plt.title('История генетического алгоритма')
        plt.legend()
        plt.grid(True)
        plt.show()

    def getResult(self) -> np.ndarray:
        historical_data = self.getHistoricalData("vec_to_model")
        return historical_data[-1] if historical_data else np.array([])