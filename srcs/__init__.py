from abc import ABC, abstractmethod
from utils.data_loader import DataLoader,DaskDataLoader,ArrowDataLoader
from utils.metric_calculator import MetricCalculator
from utils.models import Dataset, RecommendResult
import time

class BaseRecommender(ABC):
    def __init__(self, data_loader_str: str = "pandas"):
        """
        data_loader_str: 사용할 데이터 로더를 지정하는 문자열. "arrow", "pandas", "dask"
        """
        # 문자열에 따른 데이터 로더 클래스 매핑
        loader_mapping = {
            "arrow": ArrowDataLoader,
            "pandas": DataLoader,
            "dask": DaskDataLoader
        }
        #잘못 들어오면 걍 pandas 기반 dataloader 보냄
        self.data_loader_cls = loader_mapping.get(data_loader_str.lower(), DataLoader)
    
    @abstractmethod
    def recommend(self, dataset: 'Dataset', **kwargs) -> 'RecommendResult':
        """
        각 추천 알고리즘마다 구현하는 추천 메소드.
        """
        pass

    def run_sample(self,
                   num_users: int = 1000,
                   num_test_items: int = 5,
                   data_path: str = "./data/ml-10M100K/",
                   k: int = 10) -> None:
        """
        run_sample 실행 시 사용할 파라미터들을 입력받으며,
        생성자에서 설정된 데이터 로더 클래스를 사용하여 데이터를 로드한 후,
        추천 결과 평가 및 메트릭 출력.
        """
        # 데이터 로더를 통해 데이터셋 로딩
        loading_time = time.time()
        movielens = self.data_loader_cls(num_users=num_users, num_test_items=num_test_items, data_path=data_path).load()
        print(f'데이터 로딩 시간 : {time.time()-loading_time:.2f}초')
        # 추천 결과 계산
        recommend_result = self.recommend(movielens)
        
        # 추천 결과 평가
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=k,
        )
        print(metrics)