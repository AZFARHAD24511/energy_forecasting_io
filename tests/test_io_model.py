import numpy as np
from src.energy_io.io_model import LeontiefEconomicModel
from src.energy_io.data_loader import demo_io_matrix, make_demo_data


def test_demo_data_is_hourly_and_has_features():
    df = make_demo_data(periods=24 * 10)
    assert len(df) == 24 * 10 - 1
    assert {"Temperature", "WindSpeed", "Load", "Price", "Price_Lag1", "Load_Lag1"} <= set(df.columns)


def test_output_effect_shape_and_sign_for_positive_shock():
    A, sectors = demo_io_matrix()
    model = LeontiefEconomicModel(A, sectors)
    shock = np.zeros(len(sectors))
    shock[0] = 10
    result = model.output_effect(shock)
    assert len(result) == len(sectors)
    assert result.loc[0, "gross_output_change"] > 0


def test_electricity_cost_scenario():
    A, sectors = demo_io_matrix()
    model = LeontiefEconomicModel(A, sectors)
    result = model.electricity_cost_scenario("Electricity", 0.2)
    assert len(result) == len(sectors)
    assert (result["direct_cost_change_pct_points"] >= 0).all()
