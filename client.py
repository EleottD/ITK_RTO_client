import grpc
import RtoApi_pb2
import RtoApi_pb2_grpc
import numpy as np
from typing import List, Dict
from optimization_methods import SERVER_HOST, SERVER_PORT, get_logging_config

class OptimizationModel:
    def __init__(self, a=1.0):
        self.a = a

        self.curr_mv1=14.1
        self.curr_mv2=15.7
        self.curr_mv3=34.7
        self.curr_mv4=98.7
        self.curr_mv5=0.93
        self.curr_mv6=0.43

        self.curr_cv1=13.5
        self.curr_cv2=14.1
        self.curr_cv3=83.5
        self.curr_cv4=90.3
        self.curr_cv5=15.7
        self.curr_cv6=100
        self.curr_cv7=30.5
        self.curr_cv8=34.7
        self.curr_cv9=1.145
        self.curr_cv10=0.95
        self.curr_cv11=20
        self.curr_cv12=95
        self.curr_cv13=61.1
        self.curr_dv1=self.curr_mv1+self.curr_mv2+self.curr_mv3


    def evaluate(self, mv_values: List[float]) -> float:
        # Минимум: x1=2, x2=-3, x3=4, x4=1, значение 0
        x1, x2, x3, x4, x5, x6 = mv_values
        #target_func=(-50000*x5-35000*x6)
        additional_cv = [
            self.curr_cv1+0.964*(x1-self.curr_mv1),  # CV1 (индекс 1)
            self.curr_cv2+1*(x1-self.curr_mv1),   # CV2 (индекс 2)
            self.curr_cv3+1.4*(x1-self.curr_mv1), # CV3 (индекс 3)
            self.curr_cv4-0.4*(x1-self.curr_mv1), # CV4 (индекс 4)
            self.curr_cv5+1*(x2-self.curr_mv2),   # CV5 (индекс 5)
            self.curr_cv6-0.5*(x2-self.curr_mv2)+0.5*(x3-self.curr_mv3),               # CV6 (индекс 6)
            self.curr_cv7+0.88*(x3-self.curr_mv3)-2.65*(x4-self.curr_mv4),              # CV7 (индекс 7)
            self.curr_cv8+1*(x3-self.curr_mv3),               # CV8 (индекс 8)
            self.curr_cv9+0.033*(x3-self.curr_mv3)+0.01*(x4-self.curr_mv4),       # CV9 (индекс 9)
            self.curr_cv10-0.0218*(x1-self.curr_mv1)-0.015*(x2-self.curr_mv2)+0.0288*(x3-self.curr_mv3)+0.109*(x4-self.curr_mv4)+0.0655*(x5-self.curr_mv5)-0.0218*(x6-self.curr_mv6),             # CV10 (индекс 10)
            self.curr_cv11-0.546*(x1-self.curr_mv1)-0.382*(x2-self.curr_mv2)+0.721*(x3-self.curr_mv3)+0.719*(x4-self.curr_mv4)+0.764*(x5-self.curr_mv5)-0.546*(x6-self.curr_mv6),                # CV11 (индекс 11)
            self.curr_cv12-0.109*(x1-self.curr_mv1)-0.546*(x2-self.curr_mv2)+0.0546*(x3-self.curr_mv3)+0.625*(x4-self.curr_mv4)+0.371*(x5-self.curr_mv5)-0.109*(x6-self.curr_mv6),        # CV12 (индекс 12)
            self.curr_cv13+0.962*(x1-self.curr_mv1)+1*(x2-self.curr_mv2)+0.88*(x3-self.curr_mv3),         # CV13 (индекс 13)
            self.curr_dv1-x1-x2-x3
        ]
        target_func=(40000*additional_cv[13-1]-65000*x5-45000*x6)
        return [target_func]+additional_cv

class RtoClient:
    def __init__(self, server_address=None, optimization_method="GaussOpt"):
        if server_address is None:
            # Используем порт из config.json
            server_address = f'{SERVER_HOST}:{SERVER_PORT}'
        self.channel = grpc.insecure_channel(server_address)
        self.stub = RtoApi_pb2_grpc.RtoServiceStub(self.channel)
        self.model = OptimizationModel()
        self.mv_names = {}
        self.optimization_method = optimization_method
        
        # Загружаем настройки логирования
        self.logging_config = get_logging_config()
        print(f"[CLIENT] Подключение к серверу: {server_address}")

    def start_session(self) -> tuple:
        #cv = {
        #    "Id": "36127bf6-bf83-45c0-a4e1-65d2a1c20c22", "Name": "Y", "DataType": "Numeric", "LowerBound": -1e100, "UpperBound": 1e100
        #}
        cvs = [
            {
                "Id": "36127bf6-bf83-45c0-a4e1-65d2a1c20c22",
                "Name": "Target Function",
                "DataType": "Numeric",
                "LowerBound": -1e100,
                "UpperBound": 1e100
            },
            {
                "Id": "cv1", "Name": "Y1", "DataType": "Numeric", "LowerBound": 9,"UpperBound": 16.5
            },
            {
                "Id": "cv2", "Name": "Y2", "DataType": "Numeric", "LowerBound": 9.4,"UpperBound": 17.2
            },
            {
                "Id": "cv3", "Name": "Y3", "DataType": "Numeric", "LowerBound": 81.5,"UpperBound": 88.5
            },
            {
                "Id": "cv4", "Name": "Y4", "DataType": "Numeric", "LowerBound": 89,"UpperBound": 91
            },
            {
                "Id": "cv5", "Name": "Y5", "DataType": "Numeric", "LowerBound": 12.3,"UpperBound": 22.6
            },
            {
                "Id": "cv6", "Name": "Y6", "DataType": "Numeric", "LowerBound": 85,"UpperBound": 140
            },
            {
                "Id": "cv7", "Name": "Y7", "DataType": "Numeric", "LowerBound": 19.6,"UpperBound": 36
            },
            {
                "Id": "cv8", "Name": "Y8", "DataType": "Numeric", "LowerBound": 28,"UpperBound": 40
            },
            {
                "Id": "cv9", "Name": "Y9", "DataType": "Numeric", "LowerBound": 0.28,"UpperBound": 1.16
            },
            {
                "Id": "cv10", "Name": "Y10", "DataType": "Numeric", "LowerBound": 0.85,"UpperBound": 1
            },
            {
                "Id": "cv11", "Name": "Y11", "DataType": "Numeric", "LowerBound": 10,"UpperBound": 25
            },
            {
                "Id": "cv12", "Name": "Y12", "DataType": "Numeric", "LowerBound": 95,"UpperBound": 99
            },
            {
                "Id": "cv13", "Name": "Y13", "DataType": "Numeric", "LowerBound": 33.3,"UpperBound": 59.9
            },
            {
                "Id": "dv1", "Name": "material_balance", "DataType": "Numeric", "LowerBound": -0.005,"UpperBound": 0.005
            }
        ]

        mvs = [
            {"Id": "mv1", "Name": "X1", "DataType": "Numeric", "LowerBound": 9.4, "UpperBound": 17.2},
            {"Id": "mv2", "Name": "X2", "DataType": "Numeric", "LowerBound": 12.3, "UpperBound": 22.6},
            {"Id": "mv3", "Name": "X3", "DataType": "Numeric", "LowerBound": 28, "UpperBound": 40},
            {"Id": "mv4", "Name": "X4", "DataType": "Numeric", "LowerBound": 94, "UpperBound": 101},
            {"Id": "mv5", "Name": "X5", "DataType": "Numeric", "LowerBound": 0.0000001, "UpperBound": 2.8},
            {"Id": "mv6", "Name": "X6", "DataType": "Numeric", "LowerBound": 0.0000001, "UpperBound": 1.3}
        ]

        self.mv_names = {mv["Id"]: mv["Name"] for mv in mvs}

        cv_tags = [RtoApi_pb2.TagType(
            id=cv["Id"],
            name=cv["Name"],
            dataType=cv["DataType"],
            lower_bound=cv["LowerBound"],
            upper_bound=cv["UpperBound"]
        ) for cv in cvs]

        mv_tags = [
            RtoApi_pb2.TagType(
                id=mv["Id"],
                name=mv["Name"],
                dataType=mv["DataType"],
                lower_bound=mv["LowerBound"],
                upper_bound=mv["UpperBound"]
            ) for mv in mvs
        ]

        print(f"[CLIENT] Запрос на создание сессии с методом: {self.optimization_method}")
        response = self.stub.StartOptimizeSession(
            RtoApi_pb2.StartRequest(
                cvs=cv_tags,
                mvs=mv_tags,
                maximize=True,
                optimization_method=self.optimization_method,
                max_iterations=2000,  
                model_id="optimization_model"
            )
        )

        print(f"[CLIENT] Ответ сервера на создание сессии: is_good={response.is_good}, message={response.message}")
        if not response.is_good:
            raise Exception(f"Ошибка создания сессии: {response.message}")

        return (response.optimization_instance_id, [mv["Id"] for mv in mvs])
    
    def run_optimization(self, session_id: str, mv_ids: List[str]) -> Dict[str, float]:
        """Основной цикл оптимизации"""
        evaluations = 0
        cv_id = "36127bf6-bf83-45c0-a4e1-65d2a1c20c22"
        model = self.model
        max_iterations = 2000  # Сохраняем для отображения прогресса
        
        # Настройки логирования из конфигурации
        show_details = self.logging_config.get("show_iteration_details", True)
        progress_interval = self.logging_config.get("progress_report_interval", 50)
        detailed_cv = self.logging_config.get("detailed_cv_logging", True)

        last_mv_values = None

        while True:
            evaluations += 1
            if show_details:
                print(f"\n[CLIENT] ===== Итерация {evaluations}/{max_iterations} =====")
            
            # 1. Request new MV from server
            response = self.stub.OptimizeIteration(
                RtoApi_pb2.OptimizeIterationRequest(
                    optimization_instance_id=session_id,
                    mv_values=[RtoApi_pb2.TagVal(tagId=id) for id in mv_ids]
                )
            )

            mv_values = [float(tag.numericValue) for tag in response.mv_values]
            last_mv_values = mv_values
            
            if show_details:
                print(f"[CLIENT][Итерация {evaluations}] MV получены от сервера: {mv_values}")

            if len(mv_values) != len(mv_ids):
                print(f"[CLIENT][ERROR][Итерация {evaluations}] Размерность MV не совпадает с количеством MV id. Прерывание.")
                break

            # 2. Calculate all CV values (target function + constraints)
            cv = model.evaluate(mv_values)
            
            if show_details:
                print(f"[CLIENT][Итерация {evaluations}] Вычислена целевая функция: {cv[0]:.6f}")
                if detailed_cv:
                    print(f"[CLIENT][Итерация {evaluations}] Все CV: {[round(c, 4) for c in cv]}")

            # 3. Prepare data to send back
            # Target function (objective)
            objective_function = RtoApi_pb2.TagVal(
                tagId=cv_id,
                numericValue=float(cv[0]),
            )

            # All constraint CVs
            cv_values = [
                RtoApi_pb2.TagVal(
                    tagId=f"cv{i}",
                    numericValue=float(cv[i]),
                ) for i in range(1, len(cv))
            ]

            # 4. Send all CV values back to the server
            response = self.stub.OptimizeIteration(
                RtoApi_pb2.OptimizeIterationRequest(
                    optimization_instance_id=session_id,
                    mv_values=[RtoApi_pb2.TagVal(tagId=mv_ids[i], numericValue=float(mv_values[i])) for i in range(len(mv_ids))],
                    cv_values=cv_values,
                    objective_function_value=objective_function
                )
            )
            
            if show_details:
                print(f"[CLIENT][Итерация {evaluations}] Отправлены CV на сервер")

            # Check for completion
            if response.flag == 3:
                print(f"\n[CLIENT] Оптимизация завершена за {evaluations} итераций.")
                final_mv = [float(tag.numericValue) for tag in response.mv_values]
                final_cv = model.evaluate(final_mv)
                print(f"[CLIENT] Лучшая точка: {final_mv}")
                print(f"[CLIENT] Значение целевой функции: {final_cv[0]:.6f}")
                if detailed_cv:
                    for i in range(1, len(final_cv)):
                        print(f"CV{i}: {final_cv[i]:.4f}")
                return {mv_ids[i]: final_mv[i] for i in range(len(mv_ids))}

            # Показываем прогресс согласно настройкам
            if evaluations % progress_interval == 0:
                progress_percent = (evaluations / max_iterations) * 100
                print(f"[CLIENT] Прогресс: {progress_percent:.1f}% ({evaluations}/{max_iterations})")

        # If the loop exits abnormally, return the last received MV
        return {mv_ids[i]: last_mv_values[i] for i in range(len(mv_ids))}

if __name__ == "__main__":
    method = "SimulatedAnnealingOpt"
    print(f"[MAIN] Инициализация метода оптимизации: {method}")

    client = RtoClient(optimization_method=method)
    
    try:
        print("Запуск оптимизации с параметрами:")
        print("- 1 целевая функция (неограниченная)")
        print("- 13 дополнительных ограничений CV")
        print("- 6 переменных MV с ограничениями")
        print(f"- Метод оптимизации: {client.optimization_method}")
        print("- Максимальное количество итераций: 2000")
        
        session_id, mv_ids = client.start_session()
        print(f"\n[CLIENT] Создана сессия оптимизации: {session_id}")
        print(f"[CLIENT] Оптимизируемые переменные: {mv_ids}")
        
        result = client.run_optimization(session_id, mv_ids)
        
        print("\nФинальные значения оптимизированных переменных:")
        for tag_id, value in result.items():
            mv_name = client.mv_names.get(tag_id, tag_id)
            print(f"{tag_id} ({mv_name}): {value:.10f}")
        print(f"Общее количество переменных: {len(result)}")
            
    except Exception as e:
        print(f"\n[CLIENT] Ошибка в процессе оптимизации: {str(e)}")
    finally:
        client.channel.close()