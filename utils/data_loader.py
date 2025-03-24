import pandas as pd 
import os 
from typing import Tuple
from .models import Dataset

'''
add docstring
'''

class DataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "./data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        """
        MovieLens 데이터셋 전체를 로드하고 전처리하여 Dataset 객체로 반환합니다.

        - 영화 정보와 사용자 평점을 로드합니다.
        - 각 사용자에 대해 최근 평가한 `self.num_test_items`개의 영화는 테스트용으로,
        나머지는 학습용으로 분할합니다.
        - 테스트 데이터에서 평점이 4 이상인 영화만을 정답 아이템으로 간주하여,
        사용자별 정답 아이템 딕셔너리를 생성합니다.

        Returns:
            Dataset: 학습용, 테스트용, 정답 매핑, 콘텐츠 정보가 포함된 데이터셋 객체
        """
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        # ranking 용 평가 데이터는 각 사용자의 평갓값이 4 이상인 영화만을 정답으로 한다
        # 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)

    def _split_data(self, movielens: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:  #문법을 살짝 수정함
        """
        MovieLens 데이터프레임을 학습용과 테스트용으로 분할합니다.

        각 사용자에 대해 가장 최근에 평가한 영화 중 `self.num_test_items`개를 테스트용으로,
        나머지를 학습용으로 사용합니다.

        Args:
            movielens (pd.DataFrame): 사용자-영화-평가 데이터를 포함한 데이터프레임.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - 학습용 데이터프레임 (movielens_train)
                - 테스트용 데이터프레임 (movielens_test)
        """

        # 학습용과 테스트용으로 데이터를 분할한다
        # 각 사용자의 직전 5개 영화를 평가용으로 사용하고, 그 이외는 학습용으로 한다
        # 먼저, 각 사용자가 평가한 영화의 순서를 계산한다
        # 최근 부여한 영화부터 순서를 부여한다(0부터 시작)
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items] #최신 num_test_items만큼 꺼내옴
        return movielens_train, movielens_test

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]: #문법을 살짝 수정함
        """
        MovieLens 데이터셋을 로드하여 전처리된 데이터프레임을 반환합니다.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - 사용자 평점 데이터와 영화 정보가 결합된 데이터프레임 (movielens_ratings)
                - 전처리된 영화 정보 데이터프레임 (movies)
        """
        # 영화 정보 로딩(10197 작품)
        # movie_id와 제목만 사용
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"), names=m_cols, sep="::", encoding="latin-1", engine="python"
        )
        # genre를 list 형식으로 저장한다
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

        # 사용자가 부여한 영화의 태그 정보를 로딩한다
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"), names=t_cols, sep="::", engine="python"
        )
        # tag를 소문자로 한다
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # 태그 정보를 결합한다
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        # 평가 데이터를 로딩한다
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(os.path.join(self.data_path, "ratings.dat"), names=r_cols, sep="::", engine="python")

        # user 수를 num_users로 줄인다
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        # 위 데이터를 결합한다
        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies



import dask.dataframe as dd
from .models import DaskDataset

class DaskDataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "./data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        """
        MovieLens 데이터셋 전체를 로드하고 전처리하여 Dataset 객체로 반환합니다.

        - 영화 정보와 사용자 평점을 로드합니다.
        - 각 사용자에 대해 최근 평가한 `self.num_test_items`개의 영화는 테스트용으로,
        나머지는 학습용으로 분할합니다.
        - 테스트 데이터에서 평점이 4 이상인 영화만을 정답 아이템으로 간주하여,
        사용자별 정답 아이템 딕셔너리를 생성합니다.

        Returns:
            Dataset: 학습용, 테스트용, 정답 매핑, 콘텐츠 정보가 포함된 데이터셋 객체
        """
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        
        # Dask DataFrame을 Pandas로 변환하여 groupby 연산을 수행
        # 테스트 데이터는 일반적으로 작기 때문에 컴퓨테이션 비용이 크지 않음
        movielens_test_pd = movielens_test.compute()
        
        # ranking 용 평가 데이터는 각 사용자의 평갓값이 4 이상인 영화만을 정답으로 한다
        # 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
        movielens_test_user2items = (
            movielens_test_pd[movielens_test_pd.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )
        
        return DaskDataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)

    def _split_data(self, movielens: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """
        MovieLens 데이터프레임을 학습용과 테스트용으로 분할합니다.

        각 사용자에 대해 가장 최근에 평가한 영화 중 `self.num_test_items`개를 테스트용으로,
        나머지를 학습용으로 사용합니다.

        Args:
            movielens (dd.DataFrame): 사용자-영화-평가 데이터를 포함한 Dask 데이터프레임.

        Returns:
            Tuple[dd.DataFrame, dd.DataFrame]: 
                - 학습용 데이터프레임 (movielens_train)
                - 테스트용 데이터프레임 (movielens_test)
        """
        # Dask에서 사용자별 그룹핑 및 순위 지정이 복잡하므로
        # 이 부분은 파티션별로 처리
        def assign_rating_order(partition):
            partition["rating_order"] = partition.groupby("user_id")["timestamp"].rank(
                ascending=False, method="first"
            )
            return partition
        
        # 각 파티션에 적용
        movielens = movielens.map_partitions(assign_rating_order)
        
        # 분할
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        
        return movielens_train, movielens_test

    def _load(self) -> Tuple[dd.DataFrame, pd.DataFrame]:
        """
        MovieLens 데이터셋을 로드하여 전처리된 데이터프레임을 반환합니다.

        Returns:
            Tuple[dd.DataFrame, pd.DataFrame]: 
                - 사용자 평점 데이터와 영화 정보가 결합된 Dask 데이터프레임 (movielens_ratings)
                - 전처리된 영화 정보 Pandas 데이터프레임 (movies)
        """
        # 영화 정보는 작은 데이터이므로 Pandas로 처리
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"), 
            names=m_cols, 
            sep="::", 
            encoding="latin-1", 
            engine="python"
        )
        
        # genre를 list 형식으로 저장
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

        # 태그 정보 로딩 - 크기가 작으므로 Pandas 사용
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"), 
            names=t_cols, 
            sep="::", 
            engine="python"
        )
        
        # tag를 소문자로 변환
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # 태그 정보를 결합
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        # 평가 데이터는 크기가 클 수 있으므로 Dask 사용
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        dtypes = {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float64",  # 에러 메시지에 따라 float64로 지정
            "timestamp": "int64"
        }
        
        ratings = dd.read_csv(
            os.path.join(self.data_path, "ratings.dat"), 
            names=r_cols, 
            sep="::", 
            engine="python",
            blocksize="64MB",  # 적절한 블록 크기 설정
            dtype=dtypes  # 데이터 타입 명시적 지정
        )

        # user 수를 num_users로 제한
        # Dask에서 unique 및 정렬은 계산 비용이 높으므로 최적화
        # 사용자 ID의 범위를 알고 있다면 바로 적용
        # 알 수 없는 경우 샘플링하여 추정
        
        if self.num_users < 1000:  # 사용자 수가 적은 경우 계산 가능
            unique_users = ratings["user_id"].unique().compute()
            valid_user_ids = sorted(unique_users)[:self.num_users]
            max_valid_user_id = max(valid_user_ids)
            ratings = ratings[ratings.user_id <= max_valid_user_id]
        else:
            # 사용자 수가 많은 경우 효율적인 접근법
            # 가정: user_id가 1부터 순차적으로 증가한다면
            ratings = ratings[ratings.user_id <= self.num_users]

        # 영화 정보와 결합
        # Dask와 Pandas 간의 병합은 비용이 크므로 최적화
        # movies를 Dask DataFrame으로 변환하여 병합
        movies_dask = dd.from_pandas(movies, npartitions=1)
        movielens_ratings = ratings.merge(movies_dask, on="movie_id")

        return movielens_ratings, movies
    





import pyarrow as pa
import pyarrow.csv as pv
from .models import ArrowDataset
import io

class ArrowDataLoader:
    """
    pyarrow 대신 Pandas DataFrame을 활용합니다.
    파일의 구분자가 "::" 인 경우, 파일 내용을 미리 탭('\t')으로 변경하여 읽습니다.
    """
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> ArrowDataset:
        ratings_df, movie_content_df = self._load()
        train_df, test_df = self._split_data(ratings_df)
        test_user2items = test_df[test_df["rating"] >= 4].groupby("user_id")["movie_id"].apply(list).to_dict()

        return ArrowDataset(
            train=train_df,
            test=test_df,
            test_user2items=test_user2items,
            item_content=movie_content_df
        )

    def _split_data(self, movielens: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        train = movielens[movielens["rating_order"] > self.num_test_items]
        test = movielens[movielens["rating_order"] <= self.num_test_items]
        return train, test

    def _read_with_replacement(self, file_path: str, column_names: list) -> pd.DataFrame:
        # 파일을 바이너리 모드로 읽은 후, UTF-8 디코딩 및 "::"를 탭으로 대체해서 Pandas read_csv로 읽음.
        with open(file_path, "rb") as f:
            content = f.read().decode("utf-8").replace("::", "\t")
        return pd.read_csv(io.StringIO(content), sep="\t", names=column_names)

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # movies.dat 읽기
        m_cols = ["movie_id", "title", "genre"]
        movies_path = os.path.join(self.data_path, "movies.dat")
        movies = self._read_with_replacement(movies_path, m_cols)
        movies["genre"] = movies["genre"].apply(lambda x: x.split("|"))

        # tags.dat 읽기
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        tags_path = os.path.join(self.data_path, "tags.dat")
        tags = self._read_with_replacement(tags_path, t_cols)
        tags["tag"] = tags["tag"].str.lower()
        movie_tags = tags.groupby("movie_id").agg({"tag": list}).reset_index()
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        # ratings.dat 읽기
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings_path = os.path.join(self.data_path, "ratings.dat")
        ratings = self._read_with_replacement(ratings_path, r_cols)

        valid_user_ids = sorted(ratings["user_id"].unique())[:self.num_users]
        ratings = ratings[ratings["user_id"] <= max(valid_user_ids)]

        movielens_ratings = ratings.merge(movies, on="movie_id")
        return movielens_ratings, movies