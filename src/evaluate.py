import mlflow
import os
import sys

def run():
    """
    MLflow에서 최고 성능 날씨 관측 모델(기온, 습도)을 찾아 평가하고 프로덕션 배포 여부를 결정
    
    Returns:
        bool: 평가 프로세스 성공 여부
            - True: 평가 완료 (배포됨 or 배포 조건 미충족으로 배포 안함)
            - False: 평가 중 오류 발생 (실험 없음, 모델 없음, 기타 예외)
    """
    # MLflow tracking URI 설정
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    print(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    
    try:
        client = mlflow.MlflowClient()
        experiment_name = "weather_24h"
        
        # 실험 가져오기 (train.py와 동일한 방식)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"실험 '{experiment_name}'를 찾을 수 없습니다.")
            print("먼저 train.py를 실행하여 날씨 관측 모델을 훈련하고 실험을 생성해주세요.")
            
            # 디버깅: 모든 실험 목록 출력
            all_experiments = client.search_experiments()
            print(f"현재 존재하는 실험들: {[exp.name for exp in all_experiments]}")
            return False
        
        print(f"실험 ID: {exp.experiment_id}")
        
        # 기온과 습도 각각의 베스트 모델 찾기
        best_temp_run, best_humid_run = find_best_models_by_metric(client, exp.experiment_id)
        
        if best_temp_run is None or best_humid_run is None:
            print("기온 또는 습도 모델을 찾을 수 없습니다.")
            print("먼저 train.py를 실행하여 날씨 관측 모델을 훈련해주세요.")
            return False
        
        # 각 모델의 성능 추출
        best_temp_rmse = best_temp_run.data.metrics.get("rmse_temp")
        best_humid_rmse = best_humid_run.data.metrics.get("rmse_humid")
        
        if best_temp_rmse is None or best_humid_rmse is None:
            print("RMSE 메트릭을 찾을 수 없습니다.")
            return False
        
        print(f"\n📊 각 모델별 베스트 성능:")
        print(f"  기온 베스트 모델: RMSE {best_temp_rmse:.3f} (Run ID: {best_temp_run.info.run_id})")
        print(f"  습도 베스트 모델: RMSE {best_humid_rmse:.3f} (Run ID: {best_humid_run.info.run_id})")
        
        # 현재 프로덕션 모델들의 성능 조회
        prod_temp_rmse, prod_humid_rmse = get_production_model_performance(client)
        
        print(f"\n📊 성능 비교 및 배포 결정:")
        
        # 기온 모델 평가
        if prod_temp_rmse is None:
            print(f"  기온 모델: 프로덕션 모델 없음 → 새 모델 배포 필요")
            temp_should_deploy = True
        else:
            print(f"  기온 모델: 현재 프로덕션 RMSE {prod_temp_rmse:.3f} vs 베스트 모델 RMSE {best_temp_rmse:.3f}")
            temp_should_deploy = best_temp_rmse < prod_temp_rmse
            print(f"    → {'베스트 모델이 더 좋음 (배포)' if temp_should_deploy else '프로덕션 모델이 더 좋음 (배포하지 않음)'}")
        
        # 습도 모델 평가  
        if prod_humid_rmse is None:
            print(f"  습도 모델: 프로덕션 모델 없음 → 새 모델 배포 필요")
            humid_should_deploy = True
        else:
            print(f"  습도 모델: 현재 프로덕션 RMSE {prod_humid_rmse:.3f} vs 베스트 모델 RMSE {best_humid_rmse:.3f}")
            humid_should_deploy = best_humid_rmse < prod_humid_rmse
            print(f"    → {'베스트 모델이 더 좋음 (배포)' if humid_should_deploy else '프로덕션 모델이 더 좋음 (배포하지 않음)'}")
        
        # 둘 중 하나라도 배포 조건을 만족하면 배포 진행
        if temp_should_deploy or humid_should_deploy:
            print(f"\n✅ 하나 이상의 모델이 프로덕션 배포 조건 충족: 프로덕션 배포 진행")
            
            deployment_success = True
            
            # 기온 모델 배포 (조건을 만족하는 경우에만)
            if temp_should_deploy:
                try:
                    success_temp = transition_model_to_production(client, "seoul_temp", best_temp_run.info.run_id)
                    if not success_temp:
                        deployment_success = False
                except Exception as e:
                    print(f"기온 모델 프로덕션 전환 중 오류 발생: {e}")
                    deployment_success = False
            else:
                print(f"기온 모델: 프로덕션 배포 조건 미충족으로 배포하지 않음")
            
            # 습도 모델 배포 (조건을 만족하는 경우에만)
            if humid_should_deploy:
                try:
                    success_humid = transition_model_to_production(client, "seoul_humid", best_humid_run.info.run_id)
                    if not success_humid:
                        deployment_success = False
                except Exception as e:
                    print(f"습도 모델 프로덕션 전환 중 오류 발생: {e}")
                    deployment_success = False
            else:
                print(f"습도 모델: 프로덕션 배포 조건 미충족으로 배포하지 않음")
            
            if deployment_success:
                deployed_models = []
                if temp_should_deploy:
                    deployed_models.append("기온")
                if humid_should_deploy:
                    deployed_models.append("습도")
                print(f"🎉 {', '.join(deployed_models)} 예측 모델 프로덕션으로 전환 완료!")
                return True
            else:
                print(f"❌ 일부 모델의 프로덕션 전환에 실패했습니다.")
                return False
        else:
            print(f"\n✅ 모든 모델 평가 완료: 프로덕션 배포 조건 미충족으로 배포하지 않음")
            print(f"    - 기온 모델: 현재 프로덕션 모델보다 성능이 낮거나 동일함")
            print(f"    - 습도 모델: 현재 프로덕션 모델보다 성능이 낮거나 동일함")
            print(f"    → 이는 정상적인 평가 결과입니다. 기존 프로덕션 모델을 유지합니다.")
            return True  # 평가 프로세스는 성공적으로 완료됨
            
    except Exception as e:
        print(f"❌ 평가 과정에서 오류 발생: {e}")
        return False

def transition_model_to_production(client, model_name, run_id):
    """특정 모델을 프로덕션으로 전환"""
    try:
        # 등록된 모델 버전 찾기
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"등록된 모델 '{model_name}'을 찾을 수 없습니다.")
            return False
        
        # 해당 run_id와 일치하는 모델 버전 찾기
        target_version = None
        for version in model_versions:
            if version.run_id == run_id:
                target_version = version.version
                break
        
        if target_version is None:
            print(f"Run ID {run_id}에 해당하는 모델 '{model_name}' 버전을 찾을 수 없습니다.")
            return False
        
        # 프로덕션으로 전환
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production",
            archive_existing_versions=True,
        )
        
        print(f"모델 '{model_name}' 버전 {target_version}을 프로덕션으로 전환했습니다.")
        return True
        
    except Exception as e:
        print(f"모델 '{model_name}' 프로덕션 전환 중 오류 발생: {e}")
        return False

def get_best_model_info():
    """
    최고 성능 날씨 관측 모델 정보를 반환하는 유틸리티 함수
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
            "rmse_temp": best_run.data.metrics.get("rmse_temp"),
            "rmse_humid": best_run.data.metrics.get("rmse_humid"),
            "start_time": best_run.info.start_time,
            "end_time": best_run.info.end_time
        }
        
    except Exception as e:
        print(f"모델 정보 조회 중 오류 발생: {e}")
        return None

def find_best_models_by_metric(client, experiment_id):
    """
    기온과 습도 각각의 베스트 RMSE 모델을 찾는 함수
    """
    try:
        # 기온 모델 베스트 찾기 (rmse_temp 기준)
        temp_runs = client.search_runs(
            experiment_id, 
            order_by=["metrics.rmse_temp ASC"], 
            max_results=1
        )
        
        # 습도 모델 베스트 찾기 (rmse_humid 기준)
        humid_runs = client.search_runs(
            experiment_id, 
            order_by=["metrics.rmse_humid ASC"], 
            max_results=1
        )
        
        best_temp_run = temp_runs[0] if temp_runs else None
        best_humid_run = humid_runs[0] if humid_runs else None
        
        return best_temp_run, best_humid_run
        
    except Exception as e:
        print(f"베스트 모델 조회 중 오류 발생: {e}")
        return None, None

def get_production_model_performance(client):
    """
    현재 프로덕션 모델들의 성능을 조회하는 함수
    """
    try:
        prod_temp_rmse = None
        prod_humid_rmse = None
        
        # 기온 모델의 프로덕션 버전 조회
        try:
            temp_prod_versions = client.get_latest_versions("seoul_temp", stages=["Production"])
            if temp_prod_versions:
                # 프로덕션 모델의 run_id로 성능 조회
                temp_run_id = temp_prod_versions[0].run_id
                temp_run = client.get_run(temp_run_id)
                prod_temp_rmse = temp_run.data.metrics.get("rmse_temp")
                if prod_temp_rmse is None:
                    # rmse_temp가 없으면 rmse를 사용 (단일 모델인 경우)
                    prod_temp_rmse = temp_run.data.metrics.get("rmse")
        except Exception as e:
            print(f"기온 모델 프로덕션 버전 조회 중 오류: {e}")
        
        # 습도 모델의 프로덕션 버전 조회
        try:
            humid_prod_versions = client.get_latest_versions("seoul_humid", stages=["Production"])
            if humid_prod_versions:
                # 프로덕션 모델의 run_id로 성능 조회
                humid_run_id = humid_prod_versions[0].run_id
                humid_run = client.get_run(humid_run_id)
                prod_humid_rmse = humid_run.data.metrics.get("rmse_humid")
                if prod_humid_rmse is None:
                    # rmse_humid가 없으면 rmse를 사용 (단일 모델인 경우)
                    prod_humid_rmse = humid_run.data.metrics.get("rmse")
        except Exception as e:
            print(f"습도 모델 프로덕션 버전 조회 중 오류: {e}")
        
        return prod_temp_rmse, prod_humid_rmse
        
    except Exception as e:
        print(f"프로덕션 모델 성능 조회 중 오류 발생: {e}")
        return None, None

if __name__ == "__main__":
    success = run()
    # 평가 프로세스 성공 여부에 따라 exit code 설정
    # True: 평가 완료 (배포됨/안됨 모두 성공) → exit 0
    # False: 평가 중 오류 발생 → exit 1  
    sys.exit(0 if success else 1) 