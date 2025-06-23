# example_usage.py
"""
Example usage scripts for golf swing classification
"""

from train import train_model
from config import DEFAULT_PATHS


def example_basic_training():
    """
    Basic training example with default settings.
    """
    print("=== 기본 훈련 예제 ===")
    
    results = train_model(
        data_root_dir=DEFAULT_PATHS['data_root_dir'],
        model_save_dir=DEFAULT_PATHS['model_save_dir'],
        project_name='basic_training',
        weighted=True,
        batch_size=32,
        num_epochs=100
    )
    
    print(f"훈련 완료! 최고 검증 정확도: {results['best_val_acc']:.4f}")
    return results


def example_custom_training():
    """
    Custom training example with modified parameters.
    """
    print("=== 커스텀 훈련 예제 ===")
    
    # 커스텀 설정
    custom_config = {
        'data_root_dir': "path/to/your/custom/data",
        'model_save_dir': "path/to/your/custom/models",
        'project_name': 'custom_experiment',
        'weighted': True,              # 가중치 샘플링 사용
        'val_size': 0.2,              # 검증 데이터 비율
        'test_size': 0.15,            # 테스트 데이터 비율
        'batch_size': 16,             # 배치 크기
        'num_class': 5,               # 클래스 수 (예: 5개 클래스)
        'num_epochs': 150,            # 에포크 수
        'early_stop_patience': 15     # 조기 종료 patience
    }
    
    results = train_model(**custom_config)
    
    print(f"커스텀 훈련 완료!")
    print(f"최고 검증 정확도: {results['best_val_acc']:.4f}")
    print(f"최저 검증 손실: {results['best_val_loss']:.4f}")
    print(f"모델 저장 경로: {results['project_dir']}")
    
    return results


def example_multiple_experiments():
    """
    Run multiple experiments with different configurations.
    """
    print("=== 다중 실험 예제 ===")
    
    experiments = [
        {
            'name': 'exp_batch16',
            'batch_size': 16,
            'weighted': True
        },
        {
            'name': 'exp_batch32',
            'batch_size': 32,
            'weighted': True
        },
        {
            'name': 'exp_no_weighted',
            'batch_size': 32,
            'weighted': False
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n🔄 실험 시작: {exp['name']}")
        
        result = train_model(
            data_root_dir=DEFAULT_PATHS['data_root_dir'],
            model_save_dir=DEFAULT_PATHS['model_save_dir'],
            project_name=exp['name'],
            weighted=exp['weighted'],
            batch_size=exp['batch_size'],
            num_epochs=50,  # 빠른 실험을 위해 에포크 수 감소
            early_stop_patience=10
        )
        
        results[exp['name']] = result
        print(f"✅ {exp['name']} 완료: 정확도 {result['best_val_acc']:.4f}")
    
    # 결과 비교
    print("\n📊 실험 결과 비교:")
    print("-" * 50)
    for exp_name, result in results.items():
        print(f"{exp_name:15s}: {result['best_val_acc']:.4f}")
    
    return results


def example_inference_only():
    """
    Example for loading a trained model and making predictions.
    """
    print("=== 추론 전용 예제 ===")
    
    import torch
    from dataset import get_data_auto_split
    from training_utils import get_predictions
    from STGCN import Model
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Model(
        in_channels=2,
        num_class=3,
        edge_importance_weighting=True,
        graph_args={'layout': 'golf', 'strategy': 'spatial'},
        dropout=0.3
    ).to(device)
    
    # 저장된 모델 가중치 로드
    model_path = "path/to/your/trained/model.pt"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 모델 로드 완료: {model_path}")
        
        # 테스트 데이터 로드
        _, _, test_loader = get_data_auto_split(
            DEFAULT_PATHS['data_root_dir'],
            batch_size=32
        )
        
        # 예측 수행
        predictions, true_labels = get_predictions(model, test_loader, device)
        
        print(f"📊 예측 완료:")
        print(f"  - 총 샘플 수: {len(predictions)}")
        print(f"  - 예측 클래스 분포: {dict(zip(*torch.unique(torch.tensor(predictions), return_counts=True)))}")
        
        return predictions, true_labels
        
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 모델을 훈련시켜 주세요.")
        return None, None


if __name__ == "__main__":
    # 사용하고 싶은 예제의 주석을 해제하세요
    
    # 1. 기본 훈련
    # example_basic_training()
    
    # 2. 커스텀 훈련
    # example_custom_training()
    
    # 3. 다중 실험
    # example_multiple_experiments()
    
    # 4. 추론만 수행
    # example_inference_only()
    
    print("예제 스크립트입니다. 사용하고 싶은 함수의 주석을 해제하여 실행하세요.")
