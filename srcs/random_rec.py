from utils.models import RecommendResult, Dataset
from srcs import BaseRecommender
from collections import defaultdict
import numpy as np
import time

np.random.seed(0)


class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 사용자 ID와 아이템 ID에 대해 0부터 시작하는 인덱스를 할당한다
        start_time = time.time()
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # 사용자 x 아이템의 행렬에서 각 셀의 예측 평갓값은 0.5~5.0의 균등 난수로 한다
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))
        print(f"준비 단계 완료: {time.time() - start_time:.2f}초")

        rmse_start = time.time()

        # rmse 평가용으로 테스트 데이터에 나오는 사용자와 아이템의 예측 평갓값을 저장한다
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # 테스트 데이터의 아이템 ID가 학습용으로 등장하지 않는 경우도 난수를 저장한다
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # 테스트 데이터에 나타나는 사용자 ID와 아이템 ID의 인덱스를 얻어, 평갓값 행렬값을 얻는다
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results
        print(f"RMSE 평가용 예측 완료: {time.time() - rmse_start:.2f}초")
        ranking_start = time.time()

        # 순위 평가용 데이터 작성
        # 각 사용자에 대한 추천 영화는, 해당 사용자가 아직 평가하지 않은 영화 중에서 무작위로 10개 작품으로 한다
        # 키는 사용자 ID, 값은 추천 아이템의 ID 리스트
        pred_user2items = defaultdict(list)
        # 사용자가 이미 평가한 영화를 저장한다
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        
        print(f"추천 목록 생성 완료: {time.time() - ranking_start:.2f}초")
        print(f"전체 추천 시간: {time.time() - start_time:.2f}초")
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)



class My_RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        start_time = time.time()

        # 사용자 및 아이템의 인덱스 매핑 (정렬하여 고정된 순서 부여)
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = {uid: idx for idx, uid in enumerate(unique_user_ids)}
        movie_id2index = {mid: idx for idx, mid in enumerate(unique_movie_ids)}

        # 사용자 x 아이템 평점 행렬: 0.5~5.0 사이의 균등 난수로 초기화
        n_users = len(unique_user_ids)
        n_movies = len(unique_movie_ids)
        pred_matrix = np.random.uniform(0.5, 5.0, (n_users, n_movies))
        print(f"준비 단계 완료: {time.time() - start_time:.2f}초")

        # -------------------------------
        # RMSE 평가용 예측 (벡터화 버전)
        rmse_start = time.time()
        # 매핑: 존재하지 않는 경우에는 NaN 처리
        user_indices = dataset.test["user_id"].map(user_id2index)
        movie_indices = dataset.test["movie_id"].map(movie_id2index)

        # np.array 형태로 변환
        user_indices_arr = user_indices.to_numpy()
        movie_indices_arr = movie_indices.to_numpy()

        # 예측 결과를 저장할 배열 생성
        pred_scores = np.empty(len(dataset.test))

        # mask: 학습에 등장한 영화에 대해서만 인덱스가 존재
        valid_mask = ~np.isnan(movie_indices_arr) & ~np.isnan(user_indices_arr)
        valid_user_idx = user_indices_arr[valid_mask].astype(int)
        valid_movie_idx = movie_indices_arr[valid_mask].astype(int)
        pred_scores[valid_mask] = pred_matrix[valid_user_idx, valid_movie_idx]
        # 나머지 (영화 ID가 학습 데이터에 없을 경우)는 랜덤 난수 생성
        num_invalid = np.sum(~valid_mask)
        pred_scores[~valid_mask] = np.random.uniform(0.5, 5.0, num_invalid)

        movie_rating_predict = dataset.test.copy()
        movie_rating_predict["rating_pred"] = pred_scores
        print(f"RMSE 평가용 예측 완료: {time.time() - rmse_start:.2f}초")

        # -------------------------------
        # 순위 평가용 추천 목록 생성
        ranking_start = time.time()
        pred_user2items = defaultdict(list)
        # 각 사용자가 평가한 영화 목록을 집합으로 변환 (멤버십 검사 속도 향상)
        user_evaluated_movies = dataset.train.groupby("user_id")["movie_id"].apply(set).to_dict()
        
        # 각 사용자별 추천: 해당 사용자가 평가하지 않은 영화 중, 평점 예측값이 높은 상위 10개 선택
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            # 사용자가 평가한 영화의 인덱스 집합 (존재하지 않는 영화 제외)
            evaluated_indices = {movie_id2index[mid] for mid in user_evaluated_movies.get(user_id, set()) if mid in movie_id2index}
            # 해당 사용자에 대해 내림차순 정렬된 전체 영화 인덱스
            sorted_movie_indexes = np.argsort(-pred_matrix[user_index, :])
            recs = []
            for movie_index in sorted_movie_indexes:
                if movie_index in evaluated_indices:
                    continue
                recs.append(unique_movie_ids[movie_index])
                if len(recs) == 10:
                    break
            pred_user2items[user_id] = recs

        print(f"추천 목록 생성 완료: {time.time() - ranking_start:.2f}초")
        print(f"전체 추천 시간: {time.time() - start_time:.2f}초")

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    RandomRecommender(data_loader_str='pandas').run_sample()
    My_RandomRecommender(data_loader_str='arrow').run_sample()
