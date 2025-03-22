import pandas as pd 
import os 
from typing import Tuple
from .models import Dataset

'''
add docstring
'''

class DataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
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
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> DaskDataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)

        movielens_test_filtered = movielens_test[movielens_test["rating"] >= 4].compute()
        test_user2items = movielens_test_filtered.groupby("user_id")["movie_id"].apply(list).to_dict()

        return DaskDataset(movielens_train, movielens_test, test_user2items, movie_content)

    def _split_data(self, movielens: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame]:
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        return movielens_train, movielens_test

    def _load(self) -> Tuple[dd.DataFrame, dd.DataFrame]:
        m_cols = ["movie_id", "title", "genre"]
        movies = dd.read_csv(os.path.join(self.data_path, "movies.dat"), names=m_cols, sep="::", encoding="latin-1", engine="python")
        movies["genre"] = movies["genre"].map(lambda x: x.split("|"))

        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        tags = dd.read_csv(os.path.join(self.data_path, "tags.dat"), names=t_cols, sep="::", encoding="latin-1", engine="python")
        tags["tag"] = tags["tag"].str.lower()
        movie_tags = tags.groupby("movie_id").agg({"tag": list})
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = dd.read_csv(os.path.join(self.data_path, "ratings.dat"), names=r_cols, sep="::", encoding="latin-1", engine="python")
        valid_user_ids = ratings["user_id"].drop_duplicates().compute().sort_values().iloc[:self.num_users]
        ratings = ratings[ratings["user_id"].isin(valid_user_ids)]
        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies
    

import pyarrow as pa
import pyarrow.csv as pv
from .models import ArrowDataset

class ArrowDataLoader:
    '''
    pyarrow는 dask와 다르게 모든 api를 그대로 못써서 일부 연산은 pandas에 의존하도록 짜놓음
    '''
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> ArrowDataset:
        ratings_df, movie_content_df = self._load()
        train_df, test_df = self._split_data(ratings_df)

        test_user2items = test_df[test_df["rating"] >= 4].groupby("user_id")["movie_id"].apply(list).to_dict()

        return ArrowDataset(
            train=pa.Table.from_pandas(train_df),
            test=pa.Table.from_pandas(test_df),
            test_user2items=test_user2items,
            item_content=pa.Table.from_pandas(movie_content_df)
        )

    def _split_data(self, movielens: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        train = movielens[movielens["rating_order"] > self.num_test_items]
        test = movielens[movielens["rating_order"] <= self.num_test_items]
        return train, test

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        m_cols = ["movie_id", "title", "genre"]
        movies = pv.read_csv(os.path.join(self.data_path, "movies.dat"), read_options=pv.ReadOptions(column_names=m_cols), parse_options=pv.ParseOptions(delimiter="::")).to_pandas()
        movies["genre"] = movies["genre"].apply(lambda x: x.split("|"))

        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        tags = pv.read_csv(os.path.join(self.data_path, "tags.dat"), read_options=pv.ReadOptions(column_names=t_cols), parse_options=pv.ParseOptions(delimiter="::")).to_pandas()
        tags["tag"] = tags["tag"].str.lower()
        movie_tags = tags.groupby("movie_id").agg({"tag": list})
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pv.read_csv(os.path.join(self.data_path, "ratings.dat"), read_options=pv.ReadOptions(column_names=r_cols), parse_options=pv.ParseOptions(delimiter="::")).to_pandas()
        valid_user_ids = sorted(ratings["user_id"].unique())[:self.num_users]
        ratings = ratings[ratings["user_id"] <= max(valid_user_ids)]

        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies