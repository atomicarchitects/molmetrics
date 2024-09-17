import logging

import pandas as pd
from rdkit import RDLogger

log = logging.getLogger(__name__)
try:
    import molmetrics.edm_analyses.analyze as edm_analyze
except ImportError:
    raise ValueError("EDM analyses not available.")


def get_edm_analyses_results(molecules_dir: str, read_as_sdf: bool) -> pd.DataFrame:
    """Returns the EDM analyses results for the given directory as a pandas dataframe."""
    # Disable RDKit logging.
    RDLogger.DisableLog("rdApp.info")
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    metrics = edm_analyze.analyze_stability_for_molecules_in_dir(
        molecules_dir, read_as_sdf=read_as_sdf
    )
    return pd.DataFrame().from_dict(
        {"path": molecules_dir, **{key: [val] for key, val in metrics.items()}}
    )
