"""
pipeline/__init__.py
====================
Trading pipeline package.
"""

from pipeline.config import (
    LABEL_CFG, FEATURE_CFG, MODEL_CFG, WF_CFG, EVAL_CFG,
    MODEL_DIR, REPORT_DIR, DATA_DIR,
)
from pipeline.data_loader import load_data, validate_data, detect_gaps, get_data_summary
from pipeline.features import compute_features, get_feature_columns, verify_no_lookahead
from pipeline.labeling import compute_all_labels, find_optimal_barriers
from pipeline.model import walk_forward_train, save_model, get_feature_importance
from pipeline.evaluation import (
    generate_wf_report, plot_feature_importance,
    plot_probability_calibration, simulate_trades,
)
