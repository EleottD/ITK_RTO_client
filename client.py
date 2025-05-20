import grpc
import RtoApi_pb2
import RtoApi_pb2_grpc
import numpy as np
from typing import List, Dict

class ParabolaModel:
    def __init__(self, a=1.0):
        self.a = a

    def evaluate(self, mv_values: List[float]) -> float:
        # Минимум при X1=2, X2=-3, X3=4, X4=1, значение 10
        return float(
            -self.a * (mv_values[0] - 2) ** 2
            - self.a * (mv_values[1] + 3) ** 2
            - self.a * (mv_values[2] - 4) ** 2
            - self.a * (mv_values[3] - 1) ** 2
            +10
        )


class RtoClient:
    def __init__(self, server_address='localhost:5081', optimization_method="TestStepOpt"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = RtoApi_pb2_grpc.RtoServiceStub(self.channel)
        self.model = ParabolaModel()
        self.mv_names = {}
        self.optimization_method = optimization_method

    def start_session(self) -> tuple:
        cv = {
            "Id": "36127bf6-bf83-45c0-a4e1-65d2a1c20c22",
            "Name": "Y",
            "DataType": "Numeric",
            "LowerBound": -1e5,
            "UpperBound": 1e5
        }

        mvs = [
            {"Id": "mv1", "Name": "X1", "DataType": "Numeric", "LowerBound": -15, "UpperBound": 15},
            {"Id": "mv2", "Name": "X2", "DataType": "Numeric", "LowerBound": -15, "UpperBound": 15},
            {"Id": "mv3", "Name": "X3", "DataType": "Numeric", "LowerBound": -15, "UpperBound": 15},
            {"Id": "mv4", "Name": "X4", "DataType": "Numeric", "LowerBound": -15, "UpperBound": 15}
        ]

        self.mv_names = {mv["Id"]: mv["Name"] for mv in mvs}

        cv_tag = RtoApi_pb2.TagType(
            id=cv["Id"],
            name=cv["Name"],
            dataType=cv["DataType"],
            lower_bound=cv["LowerBound"],
            upper_bound=cv["UpperBound"]
        )

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
                cvs=[cv_tag],
                mvs=mv_tags,
                maximize=True,
                optimization_method=self.optimization_method,
                max_iterations=100,
                model_id="parabola_model"
            )
        )

        print(f"[CLIENT] Ответ сервера на создание сессии: is_good={response.is_good}, message={response.message}")
        if not response.is_good:
            raise Exception(f"Ошибка создания сессии: {response.message}")

        return response.optimization_instance_id, [mv["Id"] for mv in mvs]
    
    def run_optimization(self, session_id: str, mv_ids: List[str]) -> Dict[str, float]:
        """Основной цикл оптимизации"""
        evaluations = 0
        cv_id = "36127bf6-bf83-45c0-a4e1-65d2a1c20c22"
        model = ParabolaModel()

        while True:
            # 1. Запрос новых MV от сервера
            response = self.stub.OptimizeIteration(
                RtoApi_pb2.OptimizeIterationRequest(
                    optimization_instance_id=session_id,
                    mv_values=[RtoApi_pb2.TagVal(tagId=id) for id in mv_ids]
                )
            )

            mv_values = [tag.numericValue for tag in response.mv_values]
            print(f"[CLIENT] Получены MV от сервера: {mv_values}")

            # 2. Вычисляем CV по этим MV
            cv = model.evaluate(mv_values)
            print(f"[CLIENT] Вычислен CV={cv} для MV={mv_values}")

            # 3. Отправляем CV обратно на сервер
            response = self.stub.OptimizeIteration(
                RtoApi_pb2.OptimizeIterationRequest(
                    optimization_instance_id=session_id,
                    mv_values=[RtoApi_pb2.TagVal(tagId=mv_ids[i], numericValue=mv_values[i]) for i in range(len(mv_ids))],
                    objective_function_value=RtoApi_pb2.TagVal(tagId=cv_id, numericValue=cv)
                )
            )
            print(f"[CLIENT] Отправлен CV={cv} для MV={mv_values}")

            evaluations += 1

            # Проверка завершения
            if response.flag == 3:
                print(f"[CLIENT] Оптимизация завершена за {evaluations} шагов.")
                break

        return {mv_ids[i]: mv_values[i] for i in range(len(mv_ids))}

if __name__ == "__main__":
    # Явно указываем метод и выводим его
    method = "TestStepOpt"  # или "EvolutionaryOpt"
    print(f"[MAIN] Будет использоваться метод оптимизации: {method}")

    client = RtoClient(optimization_method=method)
    
    try:
        print("Запуск оптимизации с параметрами:")
        print("- 1 CV с фиксированными границами")
        print("- 4 MV параметра с границами ±10")
        print(f"- Метод: {client.optimization_method}")
        print("- Макс итераций: 1000")
        
        session_id, mv_ids = client.start_session()
        print(f"\n[CLIENT] Создана сессия: {session_id}")
        print(f"[CLIENT] MV параметры: {mv_ids}")
        
        result = client.run_optimization(session_id, mv_ids)
        
        print("\nФинальные значения:")
        for tag_id, value in result.items():
            mv_name = client.mv_names.get(tag_id, tag_id)
            print(f"{tag_id} ({mv_name}): {value:.10f}")
        print(f"Всего MV: {len(result)}")
            
    except Exception as e:
        print(f"\n[CLIENT] Ошибка в процессе оптимизации: {str(e)}")
    finally:
        client.channel.close()