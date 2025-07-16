import json
import os
import sys
from BlackBoxOptimizer.BoxOptimizer.GaussOpt.GaussOpt import GaussOpt
from BlackBoxOptimizer.BoxOptimizer.SimulatedAnnealingOptimizer.SimulatedAnnealingOptimizer import SimulatedAnnealingOptimizer
from BlackBoxOptimizer.BoxOptimizer.EvolutionaryOpt.EvolutionaryOpt import EvolutionaryOpt
from BlackBoxOptimizer.BoxOptimizer.TestStepOpt.TestStepOpt import TestStepOpt
from BlackBoxOptimizer.BoxOptimizer.Genetic_Algo.Genetic_Algo import Genetic_Algo

def load_config():
    """Загружает конфигурацию из единого config.json файла"""
    # Определяем путь к config файлу рядом с исполняемым файлом (работает и как скрипт, и как exe)
    if getattr(sys, 'frozen', False):
        # Если запущен как exe
        app_path = os.path.dirname(sys.executable)
    else:
        # Если запущен как скрипт
        app_path = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(app_path, 'config.json')
    
    # Значения по умолчанию
    default_config = {
        "server": {
            "host": "localhost",
            "port": 5089,
            "max_workers": 10
        },
        "optimization": {
            "default_method": "SimulatedAnnealingOpt",
            "max_iterations": 2000,
            "maximize": True,
            "timeout_seconds": 30
        },
        "hyperparameters": {
            "GaussOpt": {
                "seed": 15,
                "kernel_cfg": ["Matern", {"nu": 2.5}]
            },
            "SimulatedAnnealingOpt": {
                "seed": 1546,
                "initial_temp": 100.0,
                "min_temp": 1e-8,
                "cooling_rate": 0.97,
                "step_size": 0.5,
                "penalty_coef": 1e8
            },
            "EvolutionaryOpt": {
                "seed": 1546,
                "population_size": 30,
                "offspring_per_parent": 5,
                "mutation_prob": 0.3,
                "sigma_init": 0.2
            },
            "TestStepOpt": {
                "seed": 1546
            },
            "Genetic_Algo": {
                "seed": 1546,
                "population_size": 50,
                "init_mutation": 0.2,
                "min_mutation": 0.01,
                "elite_size": 10
            }
        },
        "logging": {
            "show_iteration_details": True,
            "progress_report_interval": 50,
            "detailed_cv_logging": True,
            "show_constraint_violations": True,
            "show_best_results": True
        }
    }
    
    # Пытаемся загрузить настройки из файла
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config = json.load(config_file)
                print(f"[CONFIG] Загружена конфигурация из: {config_path}")
                return config
        except Exception as e:
            print(f"[CONFIG][ERROR] Ошибка загрузки конфигурации: {e}, используется конфигурация по умолчанию")
            return default_config
    else:
        # Создаем файл конфигурации по умолчанию
        try:
            with open(config_path, 'w', encoding='utf-8') as config_file:
                json.dump(default_config, config_file, indent=2, ensure_ascii=False)
                print(f"[CONFIG] Создан файл конфигурации по умолчанию: {config_path}")
        except Exception as e:
            print(f"[CONFIG][ERROR] Не удалось создать файл конфигурации: {e}")
        
        return default_config

# Загружаем конфигурацию при импорте модуля
CONFIG = load_config()

# Configuration for server-client communication
SERVER_HOST = CONFIG["server"]["host"]
SERVER_PORT = CONFIG["server"]["port"]

# Optimization methods configuration
OPTIMIZATION_METHODS = {
    "GaussOpt": {
        "class": GaussOpt,
        "default_params": CONFIG["hyperparameters"]["GaussOpt"]
    },
    "SimulatedAnnealingOpt": {
        "class": SimulatedAnnealingOptimizer,
        "default_params": CONFIG["hyperparameters"]["SimulatedAnnealingOpt"]
    },
    "EvolutionaryOpt": {
        "class": EvolutionaryOpt,
        "default_params": CONFIG["hyperparameters"]["EvolutionaryOpt"]
    },
    "TestStepOpt": {
        "class": TestStepOpt,
        "default_params": CONFIG["hyperparameters"]["TestStepOpt"]
    },
    "Genetic_Algo": {
        "class": Genetic_Algo,
        "default_params": CONFIG["hyperparameters"]["Genetic_Algo"]
    }
}

def get_config():
    """Возвращает текущую конфигурацию"""
    return CONFIG

def reload_config():
    """Перезагружает конфигурацию из файла"""
    global CONFIG, SERVER_HOST, SERVER_PORT, OPTIMIZATION_METHODS
    CONFIG = load_config()
    SERVER_HOST = CONFIG["server"]["host"]
    SERVER_PORT = CONFIG["server"]["port"]
    
    # Обновляем параметры методов оптимизации
    for method_name in OPTIMIZATION_METHODS:
        if method_name in CONFIG["hyperparameters"]:
            OPTIMIZATION_METHODS[method_name]["default_params"] = CONFIG["hyperparameters"][method_name]
    
    print("[CONFIG] Конфигурация перезагружена")

def get_logging_config():
    """Возвращает настройки логирования"""
    return CONFIG.get("logging", {})

def get_optimization_config():
    """Возвращает общие настройки оптимизации"""
    return CONFIG.get("optimization", {})

def get_server_config():
    """Возвращает настройки сервера"""
    return CONFIG.get("server", {})

if __name__ == "__main__":
    # Тестирование загрузки конфигурации
    print("=== Тестирование конфигурации ===")
    print(f"Сервер: {SERVER_HOST}:{SERVER_PORT}")
    print(f"Доступные методы оптимизации: {list(OPTIMIZATION_METHODS.keys())}")
    print(f"Настройки оптимизации: {get_optimization_config()}")
    print(f"Настройки логирования: {get_logging_config()}")
    print(f"Настройки сервера: {get_server_config()}")
    
    # Пример вывода параметров конкретного метода
    method = "SimulatedAnnealingOpt"
    print(f"\nПараметры {method}:")
    for param, value in OPTIMIZATION_METHODS[method]["default_params"].items():
        print(f"  {param}: {value}")
