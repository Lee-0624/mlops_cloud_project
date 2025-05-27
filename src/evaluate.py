import mlflow
import os
import sys

def run():
    """
    MLflow에서 최고 성능 모델을 찾아 평가하고 프로덕션 배포 여부를 결정
    """
    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    try:
        client = mlflow.MlflowClient()
        
        # 실험 가져오기
        exp = client.get_experiment_by_name("weather_24h")
        if exp is None:
            print("실험 'weather_24h'를 찾을 수 없습니다.")
            return False
        
        print(f"실험 ID: {exp.experiment_id}")
        
        # 최고 성능 모델 찾기 (RMSE 기준 오름차순)
        runs = client.search_runs(
            exp.experiment_id, 
            order_by=["metrics.rmse ASC"], 
            max_results=1
        )
        
        if not runs:
            print("실행된 모델이 없습니다.")
            return False
        
        best_run = runs[0]
        best_rmse = best_run.data.metrics.get("rmse")
        
        if best_rmse is None:
            print("RMSE 메트릭을 찾을 수 없습니다.")
            return False
        
        print(f"최고 성능 모델 RMSE: {best_rmse}")
        print(f"Run ID: {best_run.info.run_id}")
        
        # 임계값 설정 (RMSE < 3)
        threshold = 3.0
        
        if best_rmse < threshold:
            print(f"RMSE {best_rmse} < {threshold}: 프로덕션 배포 진행")
            
            # 모델을 프로덕션으로 전환
            try:
                # 모델 버전 정보 가져오기
                model_name = "seoul_temp"
                
                # 등록된 모델 버전 찾기
                model_versions = client.search_model_versions(f"name='{model_name}'")
                
                if not model_versions:
                    print(f"등록된 모델 '{model_name}'을 찾을 수 없습니다.")
                    return False
                
                # 해당 run_id와 일치하는 모델 버전 찾기
                target_version = None
                for version in model_versions:
                    if version.run_id == best_run.info.run_id:
                        target_version = version.version
                        break
                
                if target_version is None:
                    print(f"Run ID {best_run.info.run_id}에 해당하는 모델 버전을 찾을 수 없습니다.")
                    return False
                
                # 프로덕션으로 전환
                client.transition_model_version_stage(
                    name=model_name,
                    version=target_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                
                print(f"모델 버전 {target_version}을 프로덕션으로 전환했습니다.")
                return True
                
            except Exception as e:
                print(f"모델 프로덕션 전환 중 오류 발생: {e}")
                return False
        else:
            print(f"RMSE {best_rmse} >= {threshold}: 프로덕션 배포 기준 미달")
            return False
            
    except Exception as e:
        print(f"평가 과정에서 오류 발생: {e}")
        return False

def get_best_model_info():
    """
    최고 성능 모델 정보를 반환하는 유틸리티 함수
    """
    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("weather_24h")
        
        if exp is None:
            return None
        
        runs = client.search_runs(
            exp.experiment_id, 
            order_by=["metrics.rmse ASC"], 
            max_results=1
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "rmse": best_run.data.metrics.get("rmse"),
            "start_time": best_run.info.start_time,
            "end_time": best_run.info.end_time
        }
        
    except Exception as e:
        print(f"모델 정보 조회 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    success = run()
    # 평가 결과에 따라 exit code 설정
    sys.exit(0 if success else 1) 