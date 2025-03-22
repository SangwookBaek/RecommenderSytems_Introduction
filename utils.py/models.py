import dataclasses
import pandas as pd
from typing import Dict, List
'''
frozen 옵션은 재할당을 불가하게함 e.g. A.x = 10 ->이런식의 수정 불가능'
'''


@dataclasses.dataclass(frozen=True)
# 추천 시스템의 학습과 평가에 사용하는 데이터셋
class Dataset:
    # 학습용 평갓값 데이터셋
    train: pd.DataFrame
    # 테스트용 평갓값 데이터셋
    test: pd.DataFrame
    # 순위 지표의 테스트 데이터셋. 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
    test_user2items: Dict[int, List[int]]
    # 아이템 콘텐츠 정보
    item_content: pd.DataFrame


import dask.dataframe as dd
@dataclasses.dataclass(frozen=True)
class DaskDataset:
    '''
    Dask를 사용해 최적화, 병렬처리 지원, Pandas api를 그대로 사용함
    dd.read_csv()로 불러와야함
    '''
    # 학습용 평갓값 데이터셋
    train: dd.DataFrame
    # 테스트용 평갓값 데이터셋
    test: dd.DataFrame
    # 순위 지표의 테스트 데이터셋. 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
    test_user2items: Dict[int, List[int]]
    # 아이템 콘텐츠 정보
    item_content: dd.DataFrame


import pyarrow as pa
@dataclasses.dataclass(frozen=True)
class ArrowDataset:
    '''
    pyarrow로 최적화, IO 속도가 빨라진다. column wise라 메모리 사용이 효율적
    pa.table() or pa.csv.read_csv()로 불러와야함
    '''
    # 학습용 평갓값 데이터셋
    train: pa.Table
    # 테스트용 평갓값 데이터셋
    test: pa.Table
    # 순위 지표의 테스트 데이터셋. 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
    test_user2items: Dict[int, List[int]]
    # 아이템 콘텐츠 정보
    item_content: pa.Table





@dataclasses.dataclass(frozen=True)
# 추천 시스템 평가
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    # 평가 결과는 소수 셋째 자리까지만 출력한다
    def __repr__(self):
        return f"rmse={self.rmse:.3f}, Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"
    


@dataclasses.dataclass(frozen=True)
# 추천 시스템 예측 결과
class RecommendResult:
    # 테스트 데이터셋의 예측 평갓값. RMSE 평가
    rating: pd.DataFrame
    # 키는 사용자 ID, 값은 추천 아이템 ID 리스트. 순위 지표 평가.
    user2items: Dict[int, List[int]]

