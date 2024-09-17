import pandas as pd
import numpy as np
from recommenders.evaluation import python_evaluation
import warnings
from typing import List, Union

warnings.simplefilter(action='ignore', category=FutureWarning)


DEFAULT_METRICS = [
    'ndcg_at_k',
    'hr_k',
    # 'mrr_k',
]
DEFAULT_METRICS_BEYOND_ACCURACY = [
    'catalog_coverage',
]

# Mapping of Metric Names
METRIC_NAMES = {
    'map_at_k': 'map',
    'ndcg_at_k': 'ndcg',
    'recall_at_k': 'recall',
    'precision_at_k': 'precision',
    'catalog_coverage': 'cov',
    'distributional_coverage': 'entropy',
    'hr_k': 'hr',
    'mrr_k': 'mrr'
}

class Evaluator:
    """Class for computing recommendation metrics."""

    def __init__(self, 
                 metrics: List[str] = DEFAULT_METRICS,
                 metrics_beyond_accuracy: List[str] = DEFAULT_METRICS_BEYOND_ACCURACY,
                 top_k: List[int] = [10], 
                 col_user: str = 'userid', 
                 col_item: str = 'itemid',
                 col_score: str = 'score', 
                 col_rating: str = 'rating') -> None:
        
        self.metrics = metrics
        self.metrics_beyond_accuracy = metrics_beyond_accuracy
        self.top_k = top_k
        self.col_user = col_user
        self.col_item = col_item
        self.col_score = col_score
        self.col_rating = col_rating

    def downvote_seen_items(self, scores: np.ndarray, history: pd.DataFrame) -> None:
        """Downvote the scores of items already seen by users in the past.
           Makes sense to use it for datasets where items are not repeated within users (e.g., MovieLens).
        Args:
           scores: Model scores for each user for each item in the catalog, [n_holdout_users, n_items]
           history: Dataframe with interaction histories for each holdout user. Unique userid should appear in the same order as in scores. Contain columns: col_user, col_item
        Returns:
           scores_downvoted: Downvoted model scores
        """
        scores_downvoted = scores.copy()
        user_idx = history[self.col_user].values
        item_idx = history[self.col_item].values
        user_idx, _ = pd.factorize(user_idx, sort=True)
        seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores_downvoted.shape)
        np.put(scores_downvoted, seen_idx_flat, -np.inf)

        return scores_downvoted

    def topk_recommendations(self, scores: np.ndarray) -> np.ndarray:
        """Get top k recommendations for each user based on the scores.
        Args:
            scores: Model scores for each user for each item in the catalog, [n_holdout_users, n_items]
        Returns:
            Array [n_holdout_users, max(self.top_k)] of recommendations with items sorted by model scores)
        """
        return np.apply_along_axis(self._topidx, 1, scores, max(self.top_k))

    def compute_metrics(self, 
                        holdout: pd.DataFrame, 
                        recs: Union[pd.DataFrame, np.ndarray], 
                        train: pd.DataFrame = None) -> dict:
        """Compute all specified metrics for recommendations.
        Args:
            holdout: Dataframe with holdout data (contains columns: col_user, col_item, col_rating).
            recs: Dataframe (contains columns: col_user, col_item, col_score) or array [n_holdout_users, n_recommended_items] of recommendations (items in n_recommended_items dimension assumed to be sorted by model scores).
            train: Dataframe with training data (contains columns: col_user, col_item).
        Returns:
            metrics
        """
        if isinstance(recs, np.ndarray):
            recs = self._array2datarecs(recs, holdout)
        
        recs_array = self._datarecs2array(recs)[:, :max(self.top_k)]
        n_test_users = holdout[self.col_user].nunique()

        assert n_test_users == recs[self.col_user].nunique()

        if self.col_rating not in holdout.columns:
            holdout[self.col_rating] = 1

        result = {}
        for k in self.top_k:
            result.update(self._compute_accuracy_metrics(holdout, recs, recs_array, k, n_test_users))
            if train is not None:
                result.update(self._compute_beyond_accuracy_metrics(recs_array, train, k))

        return result

    def _compute_accuracy_metrics(self, holdout: pd.DataFrame, recs: pd.DataFrame, 
                                  recs_array: np.ndarray, k: int, 
                                  n_test_users: int) -> dict:
        result = {}
        for metric in self.metrics:
            metric_name = METRIC_NAMES.get(metric, metric)
            if metric in ['hr_k', 'mrr_k']:
                result[f'{metric_name}@{k}'] = np.round(
                    self.get_custom_hr_mrr(metric, holdout, recs_array, n_test_users, k), 6)
            else:
                metric_func = getattr(python_evaluation, metric)
                result[f'{metric_name}@{k}'] = np.round(
                    metric_func(holdout, recs, k=k, col_user=self.col_user, 
                                col_item=self.col_item, col_prediction=self.col_score, 
                                col_rating=self.col_rating), 6)
        return result

    def _compute_beyond_accuracy_metrics(self, recs_array: np.ndarray, 
                                         train: pd.DataFrame, k: int) -> dict:
        result = {}
        for metric in self.metrics_beyond_accuracy:
            metric_name = METRIC_NAMES.get(metric, metric)
            if metric == 'catalog_coverage':
                result[f'{metric_name}@{k}'] = np.round(
                    np.unique(recs_array[:, :k]).size / train[self.col_item].nunique(), 6)
            else:
                metric_func = getattr(python_evaluation, metric)
                try:
                    result[f'{metric_name}@{k}'] = np.round(
                        metric_func(train, recs_array, col_user=self.col_user, col_item=self.col_item), 6)
                except Exception:
                    pass  # Handle metrics that may not apply
        return result

    def get_custom_hr_mrr(self, metric: str, holdout: pd.DataFrame, 
                          recs_array: np.ndarray, n_test_users: int, k: int) -> float:
        """Calculate custom HitRate and MRR metrics."""
        holdout_items = holdout.groupby(self.col_user)[self.col_item].apply(list).values
        holdout_array = np.full((n_test_users, max(map(len, holdout_items))), None)

        for i, items in enumerate(holdout_items):
            holdout_array[i, :len(items)] = items

        hits_mask = np.array([np.isin(recs_array[i, :k], holdout_array[i]) for i in range(recs_array.shape[0])])

        if metric == 'hr_k':
            return np.mean(hits_mask.any(axis=1))  # Hit Rate
        else:
            hit_ranks = np.where(hits_mask)[1] + 1.0  # MRR
            first_hit_ranks = np.zeros(n_test_users)
            first_hit_ranks[np.unique(np.where(hits_mask)[0])] = hit_ranks[
                np.unique(np.where(hits_mask)[0], return_index=True)[1]]
            return np.sum(1 / first_hit_ranks[first_hit_ranks > 0]) / n_test_users

    def _datarecs2array(self, recs: pd.DataFrame) -> np.ndarray:
        """Convert recommendation DataFrame to a numpy array."""
        return np.vstack(recs.sort_values(by=[self.col_user, self.col_score], 
                                          ascending=[True, False])
                         .groupby(self.col_user)[self.col_item]
                         .apply(list).values)

    def _array2datarecs(self, recs: np.ndarray, holdout: pd.DataFrame) -> pd.DataFrame:
        """Convert numpy array of recommendations to a DataFrame."""
        n_users, n_items = recs.shape
        return pd.DataFrame({
            self.col_user: np.repeat(holdout[self.col_user].unique(), n_items),
            self.col_item: recs.flatten(),
            self.col_score: np.tile(np.arange(n_items, 0, -1), n_users)
        })

    def _topidx(self, a: np.ndarray, topn: int) -> np.ndarray:
        """Get indices of the top N items for each user."""
        partitioned = np.argpartition(a, -topn)[-topn:]
        return partitioned[np.argsort(-a[partitioned])]