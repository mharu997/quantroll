"""quantroll — fast, temporally stable rolling-window tools for quant work.

Core ideas
----------
- **Stable rolling PCA**: per-window eigendecompositions matched across time
  (absolute cosine similarity), fixing eigenvector sign flips and component
  reordering so projected series are usable by downstream models.
- **Performance measures**: a vectorized, NaN-aware measure zoo.
- **Scorecards**: explainable multi-measure comparison of assets vs. a target.

Quick start
-----------
>>> from quantroll import RollingPCA, Scorecard, rolling_pca
>>> from quantroll import measures, simulate
"""

from . import simulate
from .core.align import align_sequence, match_step
from .core.rolling_embedding import RollingEmbeddingResult, rolling_embedding
from .core.rolling_kmeans import RollingKMeansResult, rolling_kmeans
from .core.rolling_pca import RollingPCAResult, rolling_pca
from .core.rolling_regimes import RollingRegimesResult, rolling_regimes
from .estimators.embedding import RollingEmbedding
from .estimators.kmeans import RollingKMeans
from .estimators.pca import RollingPCA
from .estimators.regimes import RollingRegimes
from .measures import metrics as measures
from .measures.scorecard import DEFAULT_MEASURES, Measure, Scorecard, ScorecardResult
from .select import select_n_clusters, select_n_regimes

__version__ = "0.6.0"

__all__ = [
    "RollingPCA",
    "rolling_pca",
    "RollingPCAResult",
    "RollingRegimes",
    "rolling_regimes",
    "RollingRegimesResult",
    "RollingKMeans",
    "rolling_kmeans",
    "RollingKMeansResult",
    "RollingEmbedding",
    "rolling_embedding",
    "RollingEmbeddingResult",
    "align_sequence",
    "match_step",
    "Scorecard",
    "ScorecardResult",
    "Measure",
    "DEFAULT_MEASURES",
    "measures",
    "simulate",
    "select_n_regimes",
    "select_n_clusters",
    "__version__",
]
