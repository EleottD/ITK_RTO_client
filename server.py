from concurrent import futures
from typing import Dict, List, Any, Optional
import threading
import time
import numpy as np
import grpc
import uuid
import json
import os
import sys
import traceback
from datetime import datetime

# Import necessary components from your RTO API
import RtoApi_pb2
import RtoApi_pb2_grpc
from optimization_methods import OPTIMIZATION_METHODS, SERVER_HOST, SERVER_PORT, get_logging_config, get_optimization_config, get_server_config
from BlackBoxOptimizer import Optimizer, OptimisationTypes

class OptimizationSession:
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        self.session_id = session_id
        self.session_data = session_data
        self.status = "idle"
        self.current_iteration = 0
        self.start_time = 0
        self.thread = None
        self.stop_event = threading.Event()
        self.optimizer = None
        self.current_cv = None
        self.last_mv = None
        self.cv_ready = threading.Event()
        self.optimization_started = False
        self.optimization_finished = False
        self.session_file = f"session_{session_id}.json"
        
        # Загружаем настройки логирования
        self.logging_config = get_logging_config()
        
        self._init_optimizer()
        self.init_session_file()
        
    def init_session_file(self):
        """Инициализирует файл сессии"""
        with open(self.session_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "optimization_method": self.session_data.get("optimization_method", "Unknown"),
                "MVs": self.session_data.get("MVs", []),
                "CVs": self.session_data.get("CVs", []),
                "last_iterations": []  # Храним только последние 10 итераций для отладки
            }, f, indent=2)

    def _init_optimizer(self):
        """Инициализация оптимизатора на основе выбранного метода"""
        method_name = self.session_data.get("optimization_method", "TestStepOpt")
        print(f"[SERVER] Выбран метод оптимизации: {method_name}")

        method_cfg = OPTIMIZATION_METHODS.get(method_name)
        if not method_cfg:
            raise ValueError(f"Метод оптимизации {method_name} не определен в настройках")

        opt_cls = method_cfg["class"]
        default_params = method_cfg.get("default_params", {})
        print(f"[SERVER] Параметры для метода: {default_params}")

        # Определяем тип оптимизации (минимизация или максимизация)
        is_maximize = self.session_data.get("maximize", False)
        opt_type = OptimisationTypes.maximize if is_maximize else OptimisationTypes.minimize
        print(f"[SERVER] Режим оптимизации: {'максимизация' if is_maximize else 'минимизация'}")

        # Получаем дискретные индексы (для булевых параметров)
        discrete_indices = []
        for idx, mv in enumerate(self.session_data["MVs"]):
            if mv.get("DataType") == "Boolean":
                discrete_indices.append(idx)
        
        if discrete_indices:
            print(f"[SERVER] Обнаружены дискретные параметры: {discrete_indices}")

        # СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ GAUSSOPT (из primerdlagauss.py)
        if method_name == "GaussOpt":
            print("[SERVER] Используем логику для GaussOpt из primerdlagauss.py")
            # Базовые параметры для GaussOpt (из primerdlagauss.py)
            from_model_size = len(self.session_data["CVs"])  # 14 (включая Target Function)
            
            base_params = {
                "to_model_vec_size": len(self.session_data["MVs"]),
                "from_model_vec_size": from_model_size,
                "iter_limit": self.session_data["max_iterations"],
            }
            
            print(f"[SERVER] {method_name}: from_model_vec_size = {from_model_size}")

            # Дополнительные параметры для Optimizer
            optimizer_params = {
                "external_model": self.external_model,  # Используем обычную функцию с логикой для GaussOpt
                "optimisation_type": opt_type,
                "target": None,
            }

            # Параметры метода из конфигурации (включая seed)
            method_specific_params = dict(default_params)
            print(f"[SERVER] GaussOpt не поддерживает discrete_indices, этот параметр будет передан через configure")

            # Создаем оптимизатор
            self.optimizer = Optimizer(
                optCls=opt_cls,
                **base_params,
                **optimizer_params,
                **method_specific_params
            )
            
            # Специальная настройка для GaussOpt - как в Example.py
            # Настраиваем ядро по умолчанию (Matern с nu=2.5, как в примере)
            self.optimizer.configure(kernel_cfg=('Matern', {'nu': 2.5}))
            
            # Если есть дискретные индексы, добавляем их отдельно
            if discrete_indices:
                self.optimizer.configure(discrete_indices=discrete_indices)
                print(f"[SERVER] GaussOpt: настроены дискретные индексы {discrete_indices}")
            
            print("[SERVER] GaussOpt: настроено ядро Matern с nu=2.5")
        else:
            # ОБЫЧНАЯ ЛОГИКА ДЛЯ ОСТАЛЬНЫХ МЕТОДОВ
            # Базовые параметры для всех методов
            base_params = {
                "to_model_vec_size": len(self.session_data["MVs"]),
                "from_model_vec_size": len(self.session_data["CVs"]),
                "iter_limit": self.session_data["max_iterations"],
            }

            # Дополнительные параметры для Optimizer
            optimizer_params = {
                "external_model": self.external_model,  # Передаем функцию-посредник
                "optimisation_type": opt_type,
                "target": None,
            }

            # Для остальных методов добавляем discrete_indices к параметрам
            method_specific_params = dict(default_params)
            method_specific_params["discrete_indices"] = discrete_indices

            # Создаем оптимизатор
            self.optimizer = Optimizer(
                optCls=opt_cls,
                **base_params,
                **optimizer_params,
                **method_specific_params
            )

        # Устанавливаем границы для MV (входных переменных)
        for idx, mv in enumerate(self.session_data["MVs"]):
            lb = mv.get("LowerBound", -np.inf)
            ub = mv.get("UpperBound", np.inf)
            self.optimizer.setVecItemLimit(
                idx, "to_model", 
                min=lb,
                max=ub
            )
            print(f"[SERVER] MV[{idx}] '{mv.get('Name')}': min={lb}, max={ub}")
            
            # Установка дискретных параметров, если они есть
            if mv.get("DataType") == "Boolean":
                self.optimizer.setVecItemType(idx, "bool", "to_model")
                print(f"[SERVER] MV[{idx}] '{mv.get('Name')}' установлен как булевый тип")

        # Устанавливаем границы для CV в зависимости от метода
        if method_name == "GaussOpt":
            # Для GaussOpt: точно как в primerdlagauss.py
            print("[SERVER] GaussOpt: Устанавливаем ограничения для CV (как в primerdlagauss.py)")
            # В массиве CVs: индекс 0 = Target Function, индексы 1-13 = CV1-CV13
            # Устанавливаем ограничения для CV1-CV13 (индексы from_model 1-13)
            
            for cv_idx in range(1, len(self.session_data["CVs"])):  # cv_idx: 1, 2, 3, ..., 13
                cv = self.session_data["CVs"][cv_idx]  # Получаем CV1, CV2, ..., CV13
                lb = cv.get("LowerBound", -np.inf)
                ub = cv.get("UpperBound", np.inf)
                
                self.optimizer.setVecItemLimit(
                    cv_idx, "from_model",  # Используем cv_idx напрямую (1, 2, 3, ..., 13)
                    min=lb,
                    max=ub
                )
                print(f"[SERVER] GaussOpt CV{cv_idx} '{cv.get('Name')}': устанавливаем на from_model[{cv_idx}], min={lb}, max={ub}")

            print(f"[SERVER] GaussOpt: Всего установлено {len(self.session_data['CVs'])-1} ограничений CV (индексы from_model 1-{len(self.session_data['CVs'])-1})")
        else:
            # Для остальных методов устанавливаем границы для всех CV (новая логика)
            print("[SERVER] Устанавливаем ограничения для CV в from_model со смещением (+1)")
            
            # Сначала установим "пустое" ограничение для индекса 0 (целевая функция)
            self.optimizer.setVecItemLimit(
                0, "from_model", 
                min=-np.inf,
                max=np.inf
            )
            print(f"[SERVER] CV[0] 'Target Function': min=-inf, max=inf (целевая функция)")
            
            # Затем устанавливаем ограничения для остальных CV с учетом смещения
            for idx, cv in enumerate(self.session_data["CVs"]):
                lb = cv.get("LowerBound", -np.inf)
                ub = cv.get("UpperBound", np.inf)
                # Смещение индекса: CV[i] будет на позиции i+1 в выходном векторе
                self.optimizer.setVecItemLimit(
                    idx+1, "from_model", 
                    min=lb,
                    max=ub
                )
                print(f"[SERVER] CV[{idx}] '{cv.get('Name')}': индекс смещен на {idx+1}, min={lb}, max={ub}")

    def external_model(self, mv):
        """
        Функция-посредник между оптимизатором и внешней моделью (клиентом)
        Передает MV клиенту и получает CV обратно
        Разная логика для GaussOpt и остальных методов
        """
        method_name = self.session_data.get("optimization_method", "Unknown")
        
        try:
            # Валидация входных MV
            if mv is None:
                print(f"[SERVER][ERROR] {method_name}: Получены пустые MV")
                return self._get_invalid_cv_array(method_name)
            
            if not hasattr(mv, '__len__') or len(mv) != len(self.session_data["MVs"]):
                print(f"[SERVER][ERROR] {method_name}: Неправильная размерность MV: {len(mv) if hasattr(mv, '__len__') else 'None'}, ожидалось {len(self.session_data['MVs'])}")
                return self._get_invalid_cv_array(method_name)
            
            # Проверка MV на корректность
            for i, mv_val in enumerate(mv):
                if np.isnan(mv_val) or np.isinf(mv_val):
                    mv_info = self.session_data["MVs"][i]
                    print(f"[SERVER][ERROR] {method_name}: MV[{i}] '{mv_info['Name']}' имеет некорректное значение: {mv_val}")
                    return self._get_invalid_cv_array(method_name)
            
            # Если оптимизация завершена, возвращаем неопределенное значение
            if self.optimization_finished or self.stop_event.is_set():
                # Уменьшаем количество таких сообщений - только каждое 20-е
                if not hasattr(self, '_finished_msg_count'):
                    self._finished_msg_count = 0
                self._finished_msg_count += 1
                if self._finished_msg_count % 20 == 1:
                    print(f"[SERVER][external_model] {method_name}: Optimization finished (call #{self._finished_msg_count})")
                
                return self._get_invalid_cv_array(method_name)
            
            # Сохраняем последний запрошенный MV для передачи клиенту
            self.last_mv = mv.copy()
            self.cv_ready.clear()
            
            iteration_info = f"Итерация {self.current_iteration}/{self.session_data.get('max_iterations', '?')}"
            print(f"[SERVER][external_model] {method_name}: {iteration_info} - Запрос на оценку MV: {mv}")
        
            # Ожидаем получения CV от клиента
            if not self.cv_ready.wait(timeout=30):
                print(f"[SERVER][ERROR] {method_name}: {iteration_info} - Timeout waiting for CV values")
                return self._get_invalid_cv_array(method_name)
        
            # current_cv содержит данные от клиента
            if self.current_cv is None:
                print(f"[SERVER][ERROR] {method_name}: {iteration_info} - Получены пустые CV от клиента")
                return self._get_invalid_cv_array(method_name)
        
            print(f"[SERVER][external_model] {method_name}: {iteration_info} - Получены CV: {self.current_cv}")
            
            # Логика проверки ограничений зависит от метода
            if method_name == "GaussOpt":
                # Для GaussOpt: формат [target_func, cv1, cv2, ..., cv13]
                return self._check_constraints_gauss_opt()
            else:
                # Для остальных методов: старая логика
                return self._check_constraints_other_methods()
                
        except Exception as e:
            print(f"[SERVER][ERROR] {method_name}: Критическая ошибка в external_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_invalid_cv_array(method_name)

    def _get_invalid_cv_array(self, method_name):
        """Возвращает массив CV с некорректными значениями в зависимости от метода"""
        try:
            if method_name == "GaussOpt":
                # Для GaussOpt: [target_func, cv1, cv2, ..., cv13] - 14 элементов
                num_cvs = len(self.session_data["CVs"]) - 1  # Исключаем Target Function (13 CV)
                invalid_value = float('inf') if not self.session_data.get("maximize", False) else float('-inf')
                return np.array([invalid_value] + [0.0] * num_cvs)
            else:
                # Для остальных методов: как было раньше
                invalid_value = float('inf') if not self.session_data.get("maximize", False) else float('-inf')
                return np.array([invalid_value] * len(self.session_data["CVs"]))
        except Exception as e:
            print(f"[SERVER][ERROR] Ошибка создания некорректного CV массива: {str(e)}")
            # Fallback: возвращаем минимальный массив
            invalid_value = float('inf') if not self.session_data.get("maximize", False) else float('-inf')
            return np.array([invalid_value])

    def _check_constraints_gauss_opt(self):
        """Проверка ограничений для GaussOpt"""
        within_bounds = True
        violations = []
        violation_count = 0
        total_violations = 0.0
        
        # Проверяем CV начиная с индекса 1 (cv1, cv2, ..., cv13)
        for i in range(1, len(self.current_cv)):
            cv_index = i - 1  # Индекс в массиве CVs конфигурации (пропускаем Target Function)
            if cv_index >= len(self.session_data["CVs"]) - 1:  # -1 потому что первый элемент Target Function
                break
                
            cv_val = self.current_cv[i]
            cv_info = self.session_data["CVs"][cv_index + 1]  # +1 потому что индекс 0 - Target Function
            lb = cv_info.get("LowerBound", -np.inf)
            ub = cv_info.get("UpperBound", np.inf)
            
            if cv_val < lb:
                within_bounds = False
                violation_val = lb - cv_val
                violations.append(f"CV{i} '{cv_info.get('Name')}': {cv_val:.4f} < {lb:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val
            elif cv_val > ub:
                within_bounds = False
                violation_val = cv_val - ub
                violations.append(f"CV{i} '{cv_info.get('Name')}': {cv_val:.4f} > {ub:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val
        
        # Оптимизированное логирование: детали показываем реже после 50 итераций
        current_iter = getattr(self, 'current_iteration', 0)
        show_details = self.logging_config.get("show_iteration_details", True)
        
        # Логируем детали:
        # - всегда для первых 50 итераций
        # - каждые 10 итераций после 50-й
        # - каждые 50 итераций после 200-й  
        should_log_details = (
            current_iter <= 50 or 
            (current_iter <= 200 and current_iter % 10 == 0) or
            (current_iter > 200 and current_iter % 50 == 0)
        )
        
        if within_bounds:
            if show_details and should_log_details:
                print(f"[SERVER][external_model] GaussOpt: Итерация {current_iter}/{self.session_data.get('max_iterations', '?')} - CV в пределах ограничений. Целевая функция: {self.current_cv[0]:.6f}")
        else:
            if self.logging_config.get("show_constraint_violations", True) and should_log_details:
                print(f"[SERVER][external_model] GaussOpt: Итерация {current_iter}/{self.session_data.get('max_iterations', '?')} - CV вне ограничений. Нарушено {violation_count} ограничений (сумм.: {total_violations:.4f})")
                if self.logging_config.get("detailed_cv_logging", True):
                    for v in violations:
                        print(f"  - {v}")
        
        # Возвращаем CV без модификаций - GaussOpt сам обрабатывает ограничения
        return np.array(self.current_cv)
    
    def _check_constraints_other_methods(self):
        """Проверка ограничений для остальных методов (как было раньше)"""
        within_bounds = True
        violations = []
        violation_count = 0
        total_violations = 0.0
        method_name = self.session_data.get("optimization_method", "Unknown")
        
        for i, cv_val in enumerate(self.current_cv):
            if i == 0:  # Пропускаем целевую функцию
                continue
                
            # Для остальных методов: current_cv[1] соответствует CVs[1], current_cv[2] -> CVs[2], etc.
            # НО в массиве CVs: индекс 0 = Target Function, поэтому нужно правильно сопоставить
            cv_info = self.session_data["CVs"][i]  # current_cv[1] -> CVs[1], current_cv[2] -> CVs[2], ...
            lb = cv_info.get("LowerBound", -np.inf)
            ub = cv_info.get("UpperBound", np.inf)
            
            if cv_val < lb:
                within_bounds = False
                violation_val = lb - cv_val
                violations.append(f"CV[{i}] '{cv_info.get('Name')}': {cv_val:.4f} < {lb:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val
            elif cv_val > ub:
                within_bounds = False
                violation_val = cv_val - ub
                violations.append(f"CV[{i}] '{cv_info.get('Name')}': {cv_val:.4f} > {ub:.4f} (нарушение: {violation_val:.4f})")
                violation_count += 1
                total_violations += violation_val

        # Оптимизированное логирование для других методов
        current_iter = getattr(self, 'current_iteration', 0)
        show_details = self.logging_config.get("show_iteration_details", True)
        
        # Логируем детали реже после 50 итераций
        should_log_details = (
            current_iter <= 50 or 
            (current_iter <= 200 and current_iter % 10 == 0) or
            (current_iter > 200 and current_iter % 50 == 0)
        )

        if within_bounds:
            if show_details and should_log_details:
                print(f"[SERVER][external_model] {method_name}: Итерация {current_iter}/{self.session_data.get('max_iterations', '?')} - CV в пределах ограничений. Целевая функция: {self.current_cv[0]:.6f}")
            
            # Сохраняем лучший допустимый MV и значение целевой функции
            if not hasattr(self, 'best_feasible_mv') or not hasattr(self, 'best_feasible_target'):
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
                if show_details and should_log_details:
                    print(f"[SERVER][external_model] {method_name}: Сохранен первый допустимый результат: target={self.best_feasible_target:.6f}")
            elif self.session_data.get("maximize", False) and self.current_cv[0] > self.best_feasible_target:
                # Для задачи максимизации
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
                if show_details and should_log_details:
                    print(f"[SERVER][external_model] {method_name}: Новый лучший результат (макс): target={self.best_feasible_target:.6f}")
            elif not self.session_data.get("maximize", False) and self.current_cv[0] < self.best_feasible_target:
                # Для задачи минимизации
                self.best_feasible_mv = self.last_mv.copy()
                self.best_feasible_target = self.current_cv[0]
                if show_details and should_log_details:
                    print(f"[SERVER][external_model] {method_name}: Новый лучший результат (мин): target={self.best_feasible_target:.6f}")
        else:
            if self.logging_config.get("show_constraint_violations", True) and should_log_details:
                print(f"[SERVER][external_model] {method_name}: Итерация {current_iter}/{self.session_data.get('max_iterations', '?')} - CV вне ограничений. Нарушено {violation_count} ограничений (сумм.: {total_violations:.4f})")
                if self.logging_config.get("detailed_cv_logging", True):
                    for v in violations:
                        print(f"  - {v}")

        # Возвращаем CV без модификаций - методы сами обрабатывают ограничения
        return np.array(self.current_cv)
        
    def _run_optimization(self):
        """Запуск оптимизации в отдельном потоке"""
        try:
            self.start_time = time.time()
            self.status = "running"
            
            # Запуск оптимизации - метод сам обрабатывает итерации
            self.optimizer.modelOptimize()
            
            self.optimization_finished = True
            self.status = "completed"
            
            print(f"[SERVER] Optimization completed in {time.time() - self.start_time:.2f} seconds")
            
        except Exception as e:
            self.status = "error"
            print(f"[SERVER] Error during optimization: {e}")
            import traceback
            traceback.print_exc()

    def process_iteration(self, cv_value=None):
        """
        Обработка итерации оптимизации
        Если cv_value не None, значит клиент отправляет результат оценки
        Иначе - клиент запрашивает новые значения MV
        """
        # Когда клиент предоставляет значения CV от оценки MV
        if cv_value is not None:
            self.current_cv = cv_value
            self.cv_ready.set()
            self.current_iteration += 1

            # Отслеживание идентичных итераций для обнаружения зацикливания
            if not hasattr(self, 'last_iterations'):
                self.last_iterations = []
                self.repeat_count = 0

            # Оптимизированное отслеживание зацикливания (проверяем только каждые 5 итераций)
            if self.current_iteration % 5 == 0 and self.last_mv is not None:
                mv_tuple = tuple(np.round(self.last_mv, 6))
                if len(self.last_iterations) > 0 and mv_tuple == self.last_iterations[-1]:
                    self.repeat_count += 1
                    if self.repeat_count >= 3:  # Уменьшили порог для быстрого выхода
                        print(f"[SERVER] Итерация {self.current_iteration}/{self.session_data.get('max_iterations', '?')} - Зацикливание обнаружено, применяем возмущение")
                        if self.session_data["optimization_method"] == "GaussOpt":
                            perturbation = np.random.uniform(-0.05, 0.05, len(self.last_mv))
                            perturbed_mv = self.last_mv + perturbation * np.abs(self.last_mv)
                            for i, mv in enumerate(self.session_data["MVs"]):
                                lb = mv.get("LowerBound", -np.inf)
                                ub = mv.get("UpperBound", np.inf)
                                perturbed_mv[i] = np.clip(perturbed_mv[i], lb, ub)
                            self.last_mv = perturbed_mv
                            self.repeat_count = 0
                            self.cv_ready.clear()
                else:
                    self.repeat_count = 0
                    self.last_iterations.append(mv_tuple)
                    if len(self.last_iterations) > 3:
                        self.last_iterations.pop(0)

            # Оптимизированное сохранение итераций (только каждые 10 итераций и финальную)
            should_save = (self.current_iteration % 10 == 0) or self.optimization_finished
            if should_save:
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Храним только последние 10 итераций для экономии памяти
                if "last_iterations" not in session_data:
                    session_data["last_iterations"] = []
                    
                session_data["last_iterations"].append({
                    "iteration": self.current_iteration,
                    "mv_values": self.last_mv.tolist() if isinstance(self.last_mv, np.ndarray) else self.last_mv,
                    "cv_values": self.current_cv,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Оставляем только последние 10 записей
                if len(session_data["last_iterations"]) > 10:
                    session_data["last_iterations"] = session_data["last_iterations"][-10:]
                    
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)

        # Проверка на завершение
        if self.optimization_finished or self.current_iteration >= self.session_data["max_iterations"]:
            self.optimization_finished = True
            try:
                result = None
                # Для GaussOpt всегда используем только getResult()
                if self.session_data["optimization_method"] == "GaussOpt":
                    result = self.optimizer.getResult()
                    print(f"[SERVER] GaussOpt: получен результат через getResult()")
                    if result is None or (hasattr(result, "__len__") and len(result) == 0):
                        print(f"[SERVER] GaussOpt: getResult() вернул пустой результат, используем последний MV")
                        result = self.last_mv
                # Для остальных алгоритмов логика прежняя
                elif hasattr(self, 'best_feasible_mv') and self.best_feasible_mv is not None:
                    print(f"[SERVER] Используем лучшее найденное допустимое решение")
                    result = self.best_feasible_mv
                else:
                    result = self.optimizer.getResult()
                    if result is None or (hasattr(result, "__len__") and len(result) == 0):
                        print(f"[SERVER] getResult() вернул пустой результат, используем последний MV")
                        result = self.last_mv
            except Exception as e:
                print(f"[SERVER] Ошибка при получении результата оптимизации: {e}")
                
                result = self.last_mv

            # Проверка границ результата
            try:
                if hasattr(result, '__iter__'):
                    for i, mv_val in enumerate(result):
                        if i < len(self.session_data["MVs"]):
                            lb = self.session_data["MVs"][i].get("LowerBound", -np.inf)
                            ub = self.session_data["MVs"][i].get("UpperBound", np.inf)
                            if mv_val < lb or mv_val > ub:
                                print(f"[SERVER] ПРЕДУПРЕЖДЕНИЕ: MV[{i}]={mv_val} вне границ [{lb}, {ub}]")
            except Exception as e:
                print(f"[SERVER] Ошибка при проверке границ результата: {e}")

            print(f"[SERVER] Итерация {self.current_iteration}/{self.session_data.get('max_iterations', '?')} - Optimization completed. Best MV: {result}")
            return result.tolist() if hasattr(result, "tolist") else list(result)

        # Первая итерация - запуск оптимизации в отдельном потоке
        if not self.optimization_started:
            self.optimization_started = True
            self.thread = threading.Thread(target=self._run_optimization)
            self.thread.daemon = True
            self.thread.start()
            timeout = time.time() + 10
            while self.last_mv is None and time.time() < timeout:
                time.sleep(0.01)
            if self.last_mv is None:
                raise TimeoutError("Оптимизатор не запросил первые значения MV вовремя")
            print(f"[SERVER] Начальные MV от оптимизатора: {self.last_mv}")
            return self.last_mv.tolist()

        timeout = time.time() + 1
        wait_count = 0
        while self.cv_ready.is_set() and time.time() < timeout:
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 50:
                print("[SERVER] Предупреждение: Длительное ожидание потребления CV")
        if self.cv_ready.is_set():
            print("[SERVER] Предупреждение: Оптимизатор не использовал значение CV")
            if self.session_data["optimization_method"] == "GaussOpt" and hasattr(self, 'repeat_count') and self.repeat_count > 0:
                print("[SERVER] Принудительный сброс CV_ready для GaussOpt")
                self.cv_ready.clear()
        print(f"[SERVER] Следующий MV для клиента: {self.last_mv}")
        return self.last_mv.tolist()

class RtoService(RtoApi_pb2_grpc.RtoServiceServicer):
    def __init__(self):
        """Инициализация сервиса RTO с расширенным мониторингом"""
        self.sessions = {}  # Словарь для хранения сессий оптимизации
        self.service_start_time = datetime.now()
        self.session_stats = {
            "total_created": 0,
            "total_completed": 0,
            "total_failed": 0,
            "active_sessions": 0
        }
        print(f"[SERVER] RTO Service initialized at {self.service_start_time}")
        
        # Запускаем фоновый поток для мониторинга сессий
        self._start_session_monitor()

    def _start_session_monitor(self):
        """Запуск фонового мониторинга сессий"""
        def monitor_sessions():
            while True:
                try:
                    time.sleep(60)  # Проверяем каждую минуту
                    self._cleanup_inactive_sessions()
                    self._log_session_statistics()
                except Exception as e:
                    print(f"[SERVER][ERROR] Ошибка в мониторе сессий: {e}")
        
        monitor_thread = threading.Thread(target=monitor_sessions, daemon=True)
        monitor_thread.start()
        print("[SERVER] Запущен монитор сессий")

    def _cleanup_inactive_sessions(self):
        """Очистка неактивных сессий"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            # Удаляем сессии, которые не активны более 2 часов
            if hasattr(session, 'start_time') and (current_time - session.start_time) > 7200:
                if session.status in ["completed", "stopped", "error"]:
                    sessions_to_remove.append(session_id)
                    print(f"[SERVER] Планируется удаление неактивной сессии: {session_id}")
        
        # Удаляем неактивные сессии
        for session_id in sessions_to_remove:
            try:
                session = self.sessions[session_id]
                if session.thread and session.thread.is_alive():
                    session.stop_event.set()
                    session.thread.join(timeout=5.0)
                
                # Удаляем файл сессии
                if hasattr(session, 'session_file') and os.path.exists(session.session_file):
                    os.remove(session.session_file)
                    print(f"[SERVER] Удален файл сессии: {session.session_file}")
                
                del self.sessions[session_id]
                print(f"[SERVER] Удалена неактивная сессия: {session_id}")
                
            except Exception as e:
                print(f"[SERVER][ERROR] Ошибка при удалении сессии {session_id}: {e}")

    def _log_session_statistics(self):
        """Логирование статистики сессий"""
        active_count = len(self.sessions)
        self.session_stats["active_sessions"] = active_count
        
        # Подсчитываем статистику по статусам
        status_counts = {}
        for session in self.sessions.values():
            status = session.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"[SERVER][STATS] Активных сессий: {active_count}, Статусы: {status_counts}")
        
        # Каждые 10 минут выводим подробную статистику
        if hasattr(self, '_last_detailed_log'):
            if time.time() - self._last_detailed_log < 600:  # 10 минут
                return
        
        self._last_detailed_log = time.time()
        uptime = datetime.now() - self.service_start_time
        print(f"[SERVER][DETAILED_STATS] Время работы: {uptime}")
        print(f"[SERVER][DETAILED_STATS] Общая статистика: {self.session_stats}")

    def _update_session_stats(self, event_type):
        """Обновление статистики сессий"""
        if event_type == "created":
            self.session_stats["total_created"] += 1
        elif event_type == "completed":
            self.session_stats["total_completed"] += 1
        elif event_type == "failed":
            self.session_stats["total_failed"] += 1

    def GetServiceStatus(self, request, context):
        """Возвращает расширенный статус сервиса"""
        try:
            uptime_seconds = (datetime.now() - self.service_start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            # Подсчитываем статистику по статусам сессий
            status_counts = {}
            for session in self.sessions.values():
                status = session.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Формируем детальное сообщение
            status_details = []
            status_details.append(f"Время работы: {uptime_hours:.1f} часов")
            status_details.append(f"Активных сессий: {len(self.sessions)}")
            status_details.append(f"Создано сессий: {self.session_stats['total_created']}")
            status_details.append(f"Завершено успешно: {self.session_stats['total_completed']}")
            status_details.append(f"Завершено с ошибкой: {self.session_stats['total_failed']}")
            
            if status_counts:
                status_details.append(f"Статусы: {status_counts}")
            
            message = "; ".join(status_details)
            
            # Определяем общий статус сервиса
            service_status = "active"
            if len(self.sessions) == 0:
                service_status = "idle"
            elif any(session.status == "error" for session in self.sessions.values()):
                service_status = "degraded"
            
            response = RtoApi_pb2.ServiceStatus(
                serviceId="RTO-Optimization-Service",
                status=service_status,
                serviceType="Optimization",
                user="system",
                message=message
            )
            return response
            
        except Exception as e:
            error_msg = f"Ошибка получения статуса сервиса: {str(e)}"
            print(f"[SERVER][ERROR] {error_msg}")
            return RtoApi_pb2.ServiceStatus(
                serviceId="RTO-Optimization-Service",
                status="error",
                serviceType="Optimization",
                user="system",
                message=error_msg
            )

    def StartOptimizeSession(self, request, context):
        """Запуск новой сессии оптимизации с расширенной валидацией"""
        session_id = ""
        try:
            # Валидация базовых параметров запроса
            validation_errors = self._validate_start_request(request)
            if validation_errors:
                error_msg = f"Ошибки валидации: {'; '.join(validation_errors)}"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.StartResponse(
                    optimization_instance_id="",
                    is_good=False,
                    message=error_msg
                )

            # Создаем новую сессию с уникальным ID
            session_id = str(uuid.uuid4())
            print(f"[SERVER] Генерирован ID сессии: {session_id}")
            
            # Подготавливаем данные сессии из запроса
            session_data = {
                "MVs": [],
                "CVs": [],
                "max_iterations": max(1, min(request.max_iterations, 10000)),  # Ограничиваем диапазон
                "optimization_method": request.optimization_method,
                "maximize": request.maximize
            }
            
            # Если итерации были скорректированы
            if session_data["max_iterations"] != request.max_iterations:
                print(f"[SERVER][WARNING] Количество итераций скорректировано с {request.max_iterations} до {session_data['max_iterations']}")

            # Добавляем информацию о MV (управляемых переменных) с валидацией
            for i, mv in enumerate(request.mvs):
                mv_errors = self._validate_mv_tag(mv, i)
                if mv_errors:
                    error_msg = f"Ошибки в MV[{i}]: {'; '.join(mv_errors)}"
                    print(f"[SERVER][ERROR] {error_msg}")
                    return RtoApi_pb2.StartResponse(
                        optimization_instance_id="",
                        is_good=False,
                        message=error_msg
                    )
                
                session_data["MVs"].append({
                    "Id": mv.id,
                    "Name": mv.name,
                    "DataType": mv.dataType,
                    "LowerBound": mv.lower_bound,
                    "UpperBound": mv.upper_bound,
                    "InitialValue": 0.0
                })
                print(f"[SERVER] MV[{i}] добавлен: {mv.name} ({mv.id}), границы: [{mv.lower_bound}, {mv.upper_bound}]")
            
            # Добавляем информацию о CV (контролируемых переменных) с валидацией
            for i, cv in enumerate(request.cvs):
                cv_errors = self._validate_cv_tag(cv, i)
                if cv_errors:
                    error_msg = f"Ошибки в CV[{i}]: {'; '.join(cv_errors)}"
                    print(f"[SERVER][ERROR] {error_msg}")
                    return RtoApi_pb2.StartResponse(
                        optimization_instance_id="",
                        is_good=False,
                        message=error_msg
                    )
                
                session_data["CVs"].append({
                    "Id": cv.id,
                    "Name": cv.name,
                    "DataType": cv.dataType,
                    "LowerBound": cv.lower_bound,
                    "UpperBound": cv.upper_bound
                })
                print(f"[SERVER] CV[{i}] добавлен: {cv.name} ({cv.id}), границы: [{cv.lower_bound}, {cv.upper_bound}]")
            
            # Создаем новую сессию оптимизации
            try:
                self.sessions[session_id] = OptimizationSession(session_id, session_data)
                self._update_session_stats("created")
                print(f"[SERVER] Сессия {session_id} успешно создана")
            except Exception as session_error:
                self._update_session_stats("failed")
                error_msg = f"Ошибка создания сессии оптимизации: {str(session_error)}"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.StartResponse(
                    optimization_instance_id="",
                    is_good=False,
                    message=error_msg
                )
            
            print(f"[SERVER] Запущена новая сессия оптимизации: {session_id}")
            print(f"[SERVER] Метод: {request.optimization_method}, MVs: {len(session_data['MVs'])}, CVs: {len(session_data['CVs'])}")
            print(f"[SERVER] Режим: {'максимизация' if request.maximize else 'минимизация'}, итераций: {session_data['max_iterations']}")
            
            # Обновляем статистику сессий
            self._update_session_stats("created")
            
            # Создаем успешный ответ
            response = RtoApi_pb2.StartResponse(
                optimization_instance_id=session_id,
                is_good=True,
                message=f"Сессия успешно создана. Метод: {request.optimization_method}, MVs: {len(session_data['MVs'])}, CVs: {len(session_data['CVs'])}"
            )
            return response
                
        except Exception as e:
            error_msg = f"Критическая ошибка при создании сессии: {str(e)}"
            print(f"[SERVER][CRITICAL] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Очистка частично созданной сессии
            if session_id and session_id in self.sessions:
                try:
                    del self.sessions[session_id]
                    print(f"[SERVER] Частично созданная сессия {session_id} удалена")
                except:
                    pass
            
            response = RtoApi_pb2.StartResponse(
                optimization_instance_id="",
                is_good=False,
                message=error_msg
            )
            return response

    def OptimizeIteration(self, request, context):
        """Обработка одной итерации оптимизации с расширенной валидацией"""
        session_id = ""
        try:
            # Валидация базовых параметров запроса
            if not hasattr(request, 'optimization_instance_id') or not request.optimization_instance_id:
                error_msg = "ID сессии оптимизации не указан"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=4,  # Ошибка
                    message=error_msg,
                    sessionId=""
                )
            
            session_id = request.optimization_instance_id
            
            # Проверка существования сессии
            if session_id not in self.sessions:
                error_msg = f"Сессия {session_id} не найдена или была удалена"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=4,  # Ошибка
                    message=error_msg,
                    sessionId=session_id
                )
            
            session = self.sessions[session_id]
            
            # Проверка статуса сессии
            if session.status == "error":
                error_msg = f"Сессия {session_id} находится в состоянии ошибки"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=4,  # Ошибка
                    message=error_msg,
                    sessionId=session_id
                )
            elif session.status == "stopped":
                error_msg = f"Сессия {session_id} была остановлена"
                print(f"[SERVER][WARNING] {error_msg}")
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=3,  # Завершено
                    message=error_msg,
                    sessionId=session_id
                )
            
            # Валидация данных CV, если они предоставлены
            cv_values = None
            if len(request.cv_values) > 0 or request.HasField("objective_function_value"):
                cv_validation_errors = self._validate_cv_values(request, session)
                if cv_validation_errors:
                    error_msg = f"Ошибки валидации CV: {'; '.join(cv_validation_errors)}"
                    print(f"[SERVER][ERROR] {error_msg}")
                    return RtoApi_pb2.OptimizeIterationResponse(
                        flag=4,  # Ошибка
                        message=error_msg,
                        sessionId=session_id
                    )
                
                # Правильное объединение CV: сначала целевая функция, затем ограничения
                if request.HasField("objective_function_value"):
                    obj_val = request.objective_function_value.numericValue
                    
                    # Проверка целевой функции на корректность
                    if np.isnan(obj_val) or np.isinf(obj_val):
                        error_msg = f"Целевая функция имеет некорректное значение: {obj_val}"
                        print(f"[SERVER][ERROR] {error_msg}")
                        return RtoApi_pb2.OptimizeIterationResponse(
                            flag=4,  # Ошибка
                            message=error_msg,
                            sessionId=session_id
                        )
                    
                    cv_constraint_values = [cv.numericValue for cv in request.cv_values]
                    cv_values = [obj_val] + cv_constraint_values
                    print(f"[SERVER] Получены значения CV от клиента: объективная={obj_val}, ограничения={cv_constraint_values}")
                else:
                    cv_values = [cv.numericValue for cv in request.cv_values]
                    print(f"[SERVER] Получены только значения CV-ограничений от клиента: {cv_values}")
        
            # Обработка итерации и получение новых MV
            try:
                mv_values = session.process_iteration(cv_values)
                
                # Валидация полученных MV
                if not mv_values or len(mv_values) != len(session.session_data["MVs"]):
                    error_msg = f"Некорректное количество MV: получено {len(mv_values) if mv_values else 0}, ожидалось {len(session.session_data['MVs'])}"
                    print(f"[SERVER][ERROR] {error_msg}")
                    return RtoApi_pb2.OptimizeIterationResponse(
                        flag=4,  # Ошибка
                        message=error_msg,
                        sessionId=session_id
                    )
                
                # Проверка MV на корректность значений
                for i, mv_val in enumerate(mv_values):
                    if np.isnan(mv_val) or np.isinf(mv_val):
                        mv_info = session.session_data["MVs"][i]
                        error_msg = f"MV[{i}] '{mv_info['Name']}' имеет некорректное значение: {mv_val}"
                        print(f"[SERVER][ERROR] {error_msg}")
                        return RtoApi_pb2.OptimizeIterationResponse(
                            flag=4,  # Ошибка
                            message=error_msg,
                            sessionId=session_id
                        )
                
            except Exception as process_error:
                error_msg = f"Ошибка при обработке итерации: {str(process_error)}"
                print(f"[SERVER][ERROR] {error_msg}")
                session.status = "error"
                self._update_session_stats("failed")
                return RtoApi_pb2.OptimizeIterationResponse(
                    flag=4,  # Ошибка
                    message=error_msg,
                    sessionId=session_id
                )
            
            # Определяем статус оптимизации
            flag = 0  # Успешная итерация
            status_message = "Итерация обработана успешно"
            
            if session.optimization_finished:
                flag = 3  # Оптимизация завершена
                status_message = f"Оптимизация завершена за {session.current_iteration} итераций"
                self._update_session_stats("completed")
                print(f"[SERVER] {status_message}")
            elif session.current_iteration >= session.session_data["max_iterations"]:
                flag = 3  # Достигнуто максимальное количество итераций
                status_message = f"Достигнуто максимальное количество итераций ({session.session_data['max_iterations']})"
                self._update_session_stats("completed")
                print(f"[SERVER] {status_message}")
        
            # Формирование ответа
            response = RtoApi_pb2.OptimizeIterationResponse(
                flag=flag,
                message=status_message,
                sessionId=session_id
            )
            
            # Добавление значений MV в ответ с проверкой границ
            for i, mv_val in enumerate(mv_values):
                mv_info = session.session_data["MVs"][i]
                
                # Финальная проверка границ перед отправкой
                lb = mv_info.get("LowerBound", -np.inf)
                ub = mv_info.get("UpperBound", np.inf)
                
                if mv_val < lb or mv_val > ub:
                    print(f"[SERVER][WARNING] MV[{i}] '{mv_info['Name']}' вне границ: {mv_val} не в [{lb}, {ub}]")
                    # Принудительно ограничиваем значение
                    mv_val = max(lb, min(mv_val, ub))
                    print(f"[SERVER][WARNING] MV[{i}] скорректировано до: {mv_val}")
                
                mv = RtoApi_pb2.TagVal(
                    tagId=mv_info["Id"], 
                    numericValue=float(mv_val),
                    isGood=True
                )
                response.mv_values.append(mv)
            
            print(f"[SERVER] Итерация {session.current_iteration} обработана, флаг={flag}")
            return response
        
        except Exception as e:
            error_msg = f"Критическая ошибка при обработке итерации: {str(e)}"
            print(f"[SERVER][CRITICAL] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Помечаем сессию как ошибочную, если она существует
            if session_id and session_id in self.sessions:
                self.sessions[session_id].status = "error"
            
            return RtoApi_pb2.OptimizeIterationResponse(
                flag=4,  # Ошибка
                message=error_msg,
                sessionId=session_id
            )

    def Pause(self, request, context):
        """Приостановка сессии оптимизации с расширенной валидацией"""
        try:
            # Валидация запроса
            if not hasattr(request, 'sessionId') or not request.sessionId:
                error_msg = "ID сессии не указан"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.PauseResponse(
                    message=error_msg,
                    is_paused=False
                )
            
            session_id = request.sessionId
            
            # Проверка существования сессии
            if session_id not in self.sessions:
                error_msg = f"Сессия {session_id} не найдена"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.PauseResponse(
                    message=error_msg,
                    is_paused=False
                )
            
            session = self.sessions[session_id]
            
            # Проверка текущего статуса
            if session.status == "paused":
                msg = f"Сессия {session_id} уже приостановлена"
                print(f"[SERVER][INFO] {msg}")
                return RtoApi_pb2.PauseResponse(
                    message=msg,
                    is_paused=True
                )
            elif session.status == "stopped":
                error_msg = f"Сессия {session_id} уже остановлена"
                print(f"[SERVER][WARNING] {error_msg}")
                return RtoApi_pb2.PauseResponse(
                    message=error_msg,
                    is_paused=False
                )
            elif session.status == "completed":
                error_msg = f"Сессия {session_id} уже завершена"
                print(f"[SERVER][WARNING] {error_msg}")
                return RtoApi_pb2.PauseResponse(
                    message=error_msg,
                    is_paused=False
                )
            elif session.status == "error":
                error_msg = f"Сессия {session_id} находится в состоянии ошибки"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.PauseResponse(
                    message=error_msg,
                    is_paused=False
                )
            
            # Приостанавливаем сессию
            session.status = "paused"
            success_msg = f"Сессия {session_id} успешно приостановлена"
            print(f"[SERVER] {success_msg}")
            
            return RtoApi_pb2.PauseResponse(
                message=success_msg,
                is_paused=True
            )
            
        except Exception as e:
            error_msg = f"Критическая ошибка при приостановке сессии: {str(e)}"
            print(f"[SERVER][CRITICAL] {error_msg}")
            import traceback
            traceback.print_exc()
            return RtoApi_pb2.PauseResponse(
                message=error_msg,
                is_paused=False
            )

    def Stop(self, request, context):
        """Остановка сессии оптимизации с расширенной валидацией"""
        try:
            # Валидация запроса
            if not hasattr(request, 'sessionId') or not request.sessionId:
                error_msg = "ID сессии не указан"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.StopResponse(
                    message=error_msg
                )
            
            session_id = request.sessionId
            
            # Проверка существования сессии
            if session_id not in self.sessions:
                error_msg = f"Сессия {session_id} не найдена"
                print(f"[SERVER][ERROR] {error_msg}")
                return RtoApi_pb2.StopResponse(
                    message=error_msg
                )
            
            session = self.sessions[session_id]
            
            # Проверка текущего статуса
            if session.status == "stopped":
                msg = f"Сессия {session_id} уже остановлена"
                print(f"[SERVER][INFO] {msg}")
                return RtoApi_pb2.StopResponse(
                    message=msg
                )
            elif session.status == "completed":
                msg = f"Сессия {session_id} уже завершена"
                print(f"[SERVER][INFO] {msg}")
                return RtoApi_pb2.StopResponse(
                    message=msg
                )
            
            # Останавливаем сессию
            try:
                session.stop_event.set()
                session.status = "stopped"
                session.optimization_finished = True
                
                # Ждем завершения потока оптимизации
                if session.thread and session.thread.is_alive():
                    print(f"[SERVER] Ожидание завершения потока оптимизации для сессии {session_id}")
                    session.thread.join(timeout=5.0)  # Ждем максимум 5 секунд
                    
                    if session.thread.is_alive():
                        print(f"[SERVER][WARNING] Поток оптимизации для сессии {session_id} не завершился в течение 5 секунд")
                
                success_msg = f"Сессия {session_id} успешно остановлена"
                print(f"[SERVER] {success_msg}")
                
                # Обновляем статистику сессий
                self._update_session_stats("completed")
                
                return RtoApi_pb2.StopResponse(
                    message=success_msg
                )
                
            except Exception as stop_error:
                error_msg = f"Ошибка при остановке сессии {session_id}: {str(stop_error)}"
                print(f"[SERVER][ERROR] {error_msg}")
                session.status = "error"
                return RtoApi_pb2.StopResponse(
                    message=error_msg
                )
            
        except Exception as e:
            error_msg = f"Критическая ошибка при остановке сессии: {str(e)}"
            print(f"[SERVER][CRITICAL] {error_msg}")
            import traceback
            traceback.print_exc()
            return RtoApi_pb2.StopResponse(
                message=error_msg
            )

    def _validate_start_request(self, request):
        """Валидация основных параметров запроса на создание сессии"""
        errors = []
        
        # Проверка метода оптимизации
        if not hasattr(request, 'optimization_method') or not request.optimization_method:
            errors.append("Метод оптимизации не указан")
        elif request.optimization_method not in OPTIMIZATION_METHODS:
            available_methods = list(OPTIMIZATION_METHODS.keys())
            errors.append(f"Неизвестный метод оптимизации '{request.optimization_method}'. Доступные: {available_methods}")
        
        # Проверка количества итераций
        if not hasattr(request, 'max_iterations') or request.max_iterations <= 0:
            errors.append("Количество итераций должно быть положительным числом")
        elif request.max_iterations > 10000:
            errors.append("Количество итераций не может превышать 10000")
        
        # Проверка наличия MV и CV
        if not hasattr(request, 'mvs') or len(request.mvs) == 0:
            errors.append("Должна быть определена хотя бы одна управляемая переменная (MV)")
        elif len(request.mvs) > 50:
            errors.append("Слишком много управляемых переменных (максимум 50)")
        
        if not hasattr(request, 'cvs') or len(request.cvs) == 0:
            errors.append("Должна быть определена хотя бы одна контролируемая переменная (CV)")
        elif len(request.cvs) > 100:
            errors.append("Слишком много контролируемых переменных (максимум 100)")
        
        return errors

    def _validate_mv_tag(self, mv, index):
        """Валидация параметров управляемой переменной"""
        errors = []
        
        # Проверка ID
        if not hasattr(mv, 'id') or not mv.id:
            errors.append("ID не может быть пустым")
        elif len(mv.id) > 100:
            errors.append("ID слишком длинный (максимум 100 символов)")
        
        # Проверка имени
        if not hasattr(mv, 'name') or not mv.name:
            errors.append("Имя не может быть пустым")
        elif len(mv.name) > 200:
            errors.append("Имя слишком длинное (максимум 200 символов)")
        
        # Проверка типа данных
        allowed_types = ["Numeric", "Boolean", "Integer"]
        if not hasattr(mv, 'dataType') or mv.dataType not in allowed_types:
            errors.append(f"Неподдерживаемый тип данных '{getattr(mv, 'dataType', 'None')}'. Допустимые: {allowed_types}")
        
        # Проверка границ для числовых типов
        if hasattr(mv, 'dataType') and mv.dataType in ["Numeric", "Integer"]:
            if not hasattr(mv, 'lower_bound') or not hasattr(mv, 'upper_bound'):
                errors.append("Для числовых типов должны быть указаны границы")
            elif mv.lower_bound >= mv.upper_bound:
                errors.append(f"Нижняя граница ({mv.lower_bound}) должна быть меньше верхней ({mv.upper_bound})")
            elif abs(mv.upper_bound - mv.lower_bound) < 1e-12:
                errors.append("Диапазон значений слишком мал (возможно деление на ноль)")
            
            # Проверка на бесконечные значения
            if mv.lower_bound == float('-inf') and mv.upper_bound == float('inf'):
                errors.append("Хотя бы одна из границ должна быть конечной")
            
            # Проверка на NaN
            if np.isnan(mv.lower_bound) or np.isnan(mv.upper_bound):
                errors.append("Границы не могут быть NaN")
        
        return errors

    def _validate_cv_tag(self, cv, index):
        """Валидация параметров контролируемой переменной"""
        errors = []
        
        # Проверка ID
        if not hasattr(cv, 'id') or not cv.id:
            errors.append("ID не может быть пустым")
        elif len(cv.id) > 100:
            errors.append("ID слишком длинный (максимум 100 символов)")
        
        # Проверка имени
        if not hasattr(cv, 'name') or not cv.name:
            errors.append("Имя не может быть пустым")
        elif len(cv.name) > 200:
            errors.append("Имя слишком длинное (максимум 200 символов)")
        
        # Проверка типа данных
        allowed_types = ["Numeric", "Boolean", "Integer"]
        if not hasattr(cv, 'dataType') or cv.dataType not in allowed_types:
            errors.append(f"Неподдерживаемый тип данных '{getattr(cv, 'dataType', 'None')}'. Допустимые: {allowed_types}")
        
        # Проверка границ для числовых типов
        if hasattr(cv, 'dataType') and cv.dataType in ["Numeric", "Integer"]:
            if hasattr(cv, 'lower_bound') and hasattr(cv, 'upper_bound'):
                if cv.lower_bound >= cv.upper_bound:
                    errors.append(f"Нижняя граница ({cv.lower_bound}) должна быть меньше верхней ({cv.upper_bound})")
                
                # Проверка на NaN
                if np.isnan(cv.lower_bound) or np.isnan(cv.upper_bound):
                    errors.append("Границы не могут быть NaN")
        
        return errors

    def _validate_cv_values(self, request, session):
        """Валидация значений CV, полученных от клиента"""
        errors = []
        
        # Проверка целевой функции
        if request.HasField("objective_function_value"):
            obj_val = request.objective_function_value.numericValue
            if not isinstance(obj_val, (int, float)):
                errors.append("Целевая функция должна быть числовым значением")
            elif np.isnan(obj_val):
                errors.append("Целевая функция не может быть NaN")
            elif np.isinf(obj_val):
                errors.append("Целевая функция не может быть бесконечностью")
        
        # Проверка количества CV
        expected_cv_count = len(session.session_data["CVs"]) - 1  # Исключаем целевую функцию
        actual_cv_count = len(request.cv_values)
        
        if actual_cv_count != expected_cv_count:
            errors.append(f"Неправильное количество CV: ожидалось {expected_cv_count}, получено {actual_cv_count}")
        
        # Проверка каждого CV значения
        for i, cv_val_proto in enumerate(request.cv_values):
            if not hasattr(cv_val_proto, 'numericValue'):
                errors.append(f"CV[{i}] не содержит числового значения")
                continue
                
            cv_val = cv_val_proto.numericValue
            
            # Проверка типа
            if not isinstance(cv_val, (int, float)):
                errors.append(f"CV[{i}] должно быть числовым значением, получено {type(cv_val)}")
                continue
            
            # Проверка на NaN и Inf
            if np.isnan(cv_val):
                errors.append(f"CV[{i}] не может быть NaN")
            elif np.isinf(cv_val):
                errors.append(f"CV[{i}] не может быть бесконечностью")
            
            # Проверка разумности значения (очень большие числа могут указывать на проблемы)
            if abs(cv_val) > 1e15:
                errors.append(f"CV[{i}] имеет подозрительно большое значение: {cv_val}")
        
        return errors

    def _validate_mv_values(self, mv_values, session):
        """Валидация значений MV перед отправкой клиенту"""
        errors = []
        
        if not mv_values:
            errors.append("Список MV значений пуст")
            return errors
        
        expected_mv_count = len(session.session_data["MVs"])
        actual_mv_count = len(mv_values)
        
        if actual_mv_count != expected_mv_count:
            errors.append(f"Неправильное количество MV: ожидалось {expected_mv_count}, получено {actual_mv_count}")
        
        for i, mv_val in enumerate(mv_values):
            if i >= len(session.session_data["MVs"]):
                break
                
            mv_info = session.session_data["MVs"][i]
            
            # Проверка типа
            if not isinstance(mv_val, (int, float)):
                errors.append(f"MV[{i}] '{mv_info['Name']}' должно быть числовым значением")
                continue
            
            # Проверка на NaN и Inf
            if np.isnan(mv_val):
                errors.append(f"MV[{i}] '{mv_info['Name']}' не может быть NaN")
            elif np.isinf(mv_val):
                errors.append(f"MV[{i}] '{mv_info['Name']}' не может быть бесконечностью")
            
            # Проверка границ
            lb = mv_info.get("LowerBound", -np.inf)
            ub = mv_info.get("UpperBound", np.inf)
            
            if mv_val < lb:
                errors.append(f"MV[{i}] '{mv_info['Name']}' ниже нижней границы: {mv_val} < {lb}")
            elif mv_val > ub:
                errors.append(f"MV[{i}] '{mv_info['Name']}' выше верхней границы: {mv_val} > {ub}")
        
        return errors
def serve():
    """Запуск gRPC сервера с настройкой из config.json"""
    # Получаем настройки сервера из конфигурации
    server_config = get_server_config()
    port = server_config.get("port", 5089)
    max_workers = server_config.get("max_workers", 10)
    
    print(f"[SERVER] Загружены настройки сервера из config.json:")
    print(f"[SERVER] Порт: {port}")
    print(f"[SERVER] Максимум потоков: {max_workers}")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    RtoApi_pb2_grpc.add_RtoServiceServicer_to_server(RtoService(), server)
    server_address = f'[::]:{port}'
    server.add_insecure_port(server_address)
    server.start()
    print(f"[SERVER] Сервер RTO запущен на порту {port}")
    print(f"[SERVER] Для изменения настроек отредактируйте config.json и перезапустите сервер")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
