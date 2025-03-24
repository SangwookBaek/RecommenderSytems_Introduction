import numpy as np
from sklearn.metrics import mean_squared_error
from .models import Metrics
from typing import Dict, List


class MetricCalculator:
    """
    A utility class to compute common evaluation metrics for recommender systems.

    Includes RMSE for rating prediction accuracy and Precision@K / Recall@K
    for top-K recommendation evaluation. Precision and Recall are calculated
    using macro-averaging across users.
    """

    def calc(
        self,
        true_rating: List[float],
        pred_rating: List[float],
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        """
        Calculate RMSE, Precision@K, and Recall@K.

        Args:
            true_rating (List[float]): Ground truth ratings.
            pred_rating (List[float]): Predicted ratings.
            true_user2items (Dict[int, List[int]]): Ground truth item interactions per user.
            pred_user2items (Dict[int, List[int]]): Predicted item rankings per user.
            k (int): Number of top items to consider.

        Returns:
            Metrics: A named tuple or dataclass containing RMSE, Precision@K, and Recall@K.
        """
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rmse, precision_at_k, recall_at_k)

    def _precision_at_k(self, true_items: List[int], pred_items: List[int], k: int) -> float:
        """
        Compute precision@k for a single user.

        Args:
            true_items (List[int]): Ground truth items.
            pred_items (List[int]): Predicted ranked items.
            k (int): Cutoff rank.

        Returns:
            float: Precision@k.

        Raises:
            ValueError: If k <= 0.
        """
        if k <= 0:
            raise ValueError("Parameter k must be greater than 0 for precision calculation.")
        return len(set(true_items) & set(pred_items[:k])) / k

    def _recall_at_k(self, true_items: List[int], pred_items: List[int], k: int) -> float:
        """
        Compute recall@k for a single user.

        Args:
            true_items (List[int]): Ground truth items.
            pred_items (List[int]): Predicted ranked items.
            k (int): Cutoff rank.

        Returns:
            float: Recall@k.

        Raises:
            ValueError: If k <= 0 or true_items is empty.
        """
        if k <= 0:
            raise ValueError("Parameter k must be greater than 0 for recall calculation.")
        if not true_items:
            raise ValueError("true_items must not be empty for recall calculation.")
        return len(set(true_items) & set(pred_items[:k])) / len(true_items)

    def _calc_rmse(self, true_rating: List[float], pred_rating: List[float]) -> float:
        """
        Compute Root Mean Square Error (RMSE) between true and predicted ratings.

        Args:
            true_rating (List[float]): Ground truth ratings.
            pred_rating (List[float]): Predicted ratings.

        Returns:
            float: RMSE score.
        """
        return np.sqrt(mean_squared_error(true_rating, pred_rating))

    def _calc_precision_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        """
        Compute macro-averaged Precision@k across all users.

        Args:
            true_user2items (Dict[int, List[int]]): Ground truth items per user.
            pred_user2items (Dict[int, List[int]]): Predicted ranked items per user.
            k (int): Cutoff rank.

        Returns:
            float: Macro-averaged Precision@k.
        """
        scores = []
        for user_id in true_user2items:
            p_at_k = self._precision_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(p_at_k)
        return np.mean(scores)

    def _calc_recall_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        """
        Compute macro-averaged Recall@k across all users.

        Args:
            true_user2items (Dict[int, List[int]]): Ground truth items per user.
            pred_user2items (Dict[int, List[int]]): Predicted ranked items per user.
            k (int): Cutoff rank.

        Returns:
            float: Macro-averaged Recall@k.
        """
        scores = []
        for user_id in true_user2items:
            r_at_k = self._recall_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(r_at_k)
        return np.mean(scores)
    


    def _calc_micro_precision_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        """
        Compute micro-averaged Precision@k across all users.

        Args:
            true_user2items (Dict[int, List[int]]): Ground truth items per user.
            pred_user2items (Dict[int, List[int]]): Predicted ranked items per user.
            k (int): Cutoff rank.

        Returns:
            float: Micro-averaged Precision@k.
        """
        total_hits = 0
        total_predicted = 0

        for user_id in true_user2items:
            true_items = set(true_user2items[user_id])
            pred_items = pred_user2items[user_id][:k]

            total_hits += len(set(pred_items) & true_items)
            total_predicted += min(k, len(pred_items))

        return total_hits / total_predicted if total_predicted > 0 else 0.0

    def _calc_micro_recall_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        """
        Compute micro-averaged Recall@k across all users.

        Args:
            true_user2items (Dict[int, List[int]]): Ground truth items per user.
            pred_user2items (Dict[int, List[int]]): Predicted ranked items per user.
            k (int): Cutoff rank.

        Returns:
            float: Micro-averaged Recall@k.
        """
        total_hits = 0
        total_true = 0

        for user_id in true_user2items:
            true_items = set(true_user2items[user_id])
            pred_items = pred_user2items[user_id][:k]

            total_hits += len(set(pred_items) & true_items)
            total_true += len(true_items)

        return total_hits / total_true if total_true > 0 else 0.0