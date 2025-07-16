import grpc
import RtoApi_pb2
import RtoApi_pb2_grpc
import numpy as np
from typing import List, Dict
from optimization_methods import SERVER_HOST, SERVER_PORT

# Задайте порт вручную здесь - он переопределит значение из optimization_methods
CUSTOM_PORT = 5089  # Измените на нужный вам порт

class OptimizationModel:
    def __init__(self, a=1.0):
        self.a = a

        self.curr_mv1=10

        self.curr_cv1=10


    def evaluate(self, mv_values: List[float]) -> float:
        # Минимум: x1=2, x2=-3, x3=4, x4=1, значение 0
        x1, x2 = mv_values
        #target_func=(-50000*x5-35000*x6)
        additional_cv = [
            x1,  # CV1 (индекс 1)
        ]
        target_func=2*(x1-3)**2-8
        return [target_func]+additional_cv

class RtoClient:
    def __init__(self, server_address=None, optimization_method="GaussOpt"):
        if server_address is None:
            # Используем порт из CUSTOM_PORT вместо порта по умолчанию
            server_address = f'{SERVER_HOST}:{CUSTOM_PORT}'
        self.channel = grpc.insecure_channel(server_address)
        self.stub = RtoApi_pb2_grpc.RtoServiceStub(self.channel)
        self.model = OptimizationModel()
        self.mv_names = {}
        self.optimization_method = optimization_method

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
                "Id": "cv1", "Name": "Y1", "DataType": "Numeric", "LowerBound": -1e100,"UpperBound": 1e100
            },
        ]

        mvs = [
            {"Id": "mv1", "Name": "X1", "DataType": "Numeric", "LowerBound": -1e100, "UpperBound": 1e100},
            {"Id": "mv2", "Name": "X2", "DataType": "Numeric", "LowerBound": -1e100, "UpperBound": 1e100},
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
                maximize=False,
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

        last_mv_values = None

        while True:
            # 1. Request new MV from server
            response = self.stub.OptimizeIteration(
                RtoApi_pb2.OptimizeIterationRequest(
                    optimization_instance_id=session_id,
                    mv_values=[RtoApi_pb2.TagVal(tagId=id) for id in mv_ids]
                )
            )

            mv_values = [float(tag.numericValue) for tag in response.mv_values]
            last_mv_values = mv_values
            print(f"[CLIENT][DEBUG] MV order (ids): {mv_ids}")
            print(f"[CLIENT][DEBUG] MV values received from server: {mv_values}")

            if len(mv_values) != len(mv_ids):
                print(f"[CLIENT][ERROR] Размерность MV не совпадает с количеством MV id! Прерывание.")
                break

            # 2. Calculate all CV values (target function + constraints)
            cv = model.evaluate(mv_values)
            print(f"[CLIENT][DEBUG] Calculated Target value: {cv[0]} for MV: {mv_values}")
            print(f"[CLIENT][DEBUG] All CVs: {cv}")

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
            print(f"[CLIENT] Sent CV={cv} for MV={mv_values}")

            evaluations += 1

            # Check for completion
            if response.flag == 3:
                print(f"[CLIENT] Optimization completed in {evaluations} steps.")
                final_mv = [float(tag.numericValue) for tag in response.mv_values]
                final_cv = model.evaluate(final_mv)
                print(f"[CLIENT] Best point (according to optimizer): {final_mv}")
                print(f"[CLIENT] Target value at best point: {final_cv[0]:.2f}")
                for i in range(1, len(final_cv)):
                    print(f"CV{i}: {final_cv[i]:.2f}")
                return {mv_ids[i]: final_mv[i] for i in range(len(mv_ids))}

        # If the loop exits abnormally, return the last received MV
        return {mv_ids[i]: last_mv_values[i] for i in range(len(mv_ids))}

if __name__ == "__main__":
    method = "SimulatedAnnealingOpt"
    print(f"[MAIN] Будет использоваться метод оптимизации: {method}")

    client = RtoClient(optimization_method=method)
    
    try:
        print("Starting optimization with:")
        print("- 1 target function (unbounded)")
        print("- 13 additional CVs with bounds")
        print("- 6 MVs with bounds")
        print(f"- Метод: {client.optimization_method}")
        print("- Макс итераций: 100")
        
        session_id, mv_ids = client.start_session()
        print(f"\n[CLIENT] Создана сессия: {session_id}")
        print(f"[CLIENT] MV параметры: {mv_ids}")
        #print(f"[CLIENT] CV параметры: {cv_ids}")
        
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