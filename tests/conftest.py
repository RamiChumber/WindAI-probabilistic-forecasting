import pytest
from windai.utils.synthetic import make_synthetic_dataset
from windai.data.splits import training_rows, evaluation_rows


# Synthetic stand-in for the model table: Jan-May 2023, i.e. 90 train issues (88 after the purge at the val
# boundary) and 32 val issues, 4 zones, 23-25 hours each. Models are fitted and scored as in the "val" fit.
@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_dataset(n_issues=122, start="2023-01-01", seed=1)


@pytest.fixture(scope="session")
def train_df(synthetic_df):
    return training_rows(synthetic_df, "val")


@pytest.fixture(scope="session")
def test_df(synthetic_df):
    return evaluation_rows(synthetic_df, "val")
