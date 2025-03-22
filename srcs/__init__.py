from abc import ABC, abstractmethod
from .utils.data_loader import DataLoader
from .utils.metric_calculator import MetricCalculator
from .utils.models import Dataset, RecommendResult


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        '''
        will be placeholder for each method.
        '''
        pass

    def run_sample(self,
                   num_users = 1000,
                   num_test_items = 5,
                   data_path = "../data/ml-10M100K/",
                   k =10
        ) -> None:
        '''
        run sample 시 파라미터 고정이 아쉬움 , 수정하자 입력으로 받아서 조절가능하게 수정함
        그리고 metric의 K도 수정가능하도록 수정함
        '''
        # Movielens 데이터 취득
        movielens = DataLoader(num_users=num_users, num_test_items=num_test_items, data_path=data_path).load()
        # 추천 계산
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
