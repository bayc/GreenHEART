import numpy as np
import pytest
import openmdao.api as om
from pytest import approx, fixture

from h2integrate.converters.heat.heat_pump import HeatPumpCostModel, HeatPumpPerformanceModel


@fixture
def plant_config():
    return {
        "plant": {
            "plant_life": 30,
            "simulation": {"n_timesteps": 24, "dt": 3600},
        },
    }


def _make_perf_config(**overrides):
    params = {
        "hp_mode": "fixed_cop",
        "system_capacity_mw_th": 2.0,
        "delivery_temp_C": 70.0,
        "cop": 4.0,
    }
    params.update(overrides)
    return {"model_inputs": {"performance_parameters": params}}


def _make_cost_config(**overrides):
    params = {
        "cost_year": 2022,
        "system_capacity_mw_th": 2.0,
        "capex_per_mw_th": 500_000.0,
        "fixed_opex_per_mw_th_per_year": 15_000.0,
        "variable_opex_per_mwh_th": 2.0,
    }
    params.update(overrides)
    return {"model_inputs": {"cost_parameters": params}}


@pytest.mark.unit
class TestHeatPumpPerformanceModel:
    def _build(self, plant_config, tech_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "hp",
            HeatPumpPerformanceModel(plant_config=plant_config, tech_config=tech_config),
            promotes=["*"],
        )
        prob.setup()
        return prob

    def test_fixed_cop_capacity_limited(self, plant_config):
        # Ample heat_in and electricity_in -> capacity-limited delivery at 2 MW,
        # elec use = 2 / cop = 0.5 MW.
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.5))
        assert prob.get_val("cop_actual") == approx(np.full(24, 4.0))
        assert prob.get_val("heat_supply_temp_C", units="degC") == approx(70.0)

    def test_fixed_cop_source_limited(self, plant_config):
        # heat_in = 1.5 MW -> delivered = heat_in / (1 - 1/cop) = 1.5 / 0.75 = 2.0
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 1.5), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))

    def test_fixed_cop_electricity_limited(self, plant_config):
        # electricity_in = 0.25 MW, cop = 4 -> delivered = 0.25 * 4 = 1.0 MW
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 10.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 0.25), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.25))

    def test_carnot_cop_matches_theory(self, plant_config):
        # COP = eta * T_sink / (T_sink - T_source); sink=70C, source=30C, eta=0.5
        cop_expected = 0.5 * (70 + 273.15) / ((70 + 273.15) - (30 + 273.15))
        prob = self._build(
            plant_config,
            _make_perf_config(hp_mode="carnot", cop=None, carnot_efficiency=0.5),
        )
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("cop_actual") == approx(np.full(24, cop_expected), rel=1e-6)
        # Delivered is capacity-limited at 2 MW; electricity used = 2 / cop.
        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))
        assert prob.get_val("electricity_used", units="MW") == approx(
            np.full(24, 2.0 / cop_expected), rel=1e-6
        )

    def test_carnot_infeasible_when_source_above_sink(self, plant_config):
        prob = self._build(
            plant_config,
            _make_perf_config(hp_mode="carnot", cop=None, carnot_efficiency=0.5),
        )
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 90.0, units="degC")  # above delivery
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.zeros(24))
        assert prob.get_val("cop_actual") == approx(np.zeros(24))

    def test_fixed_cop_requires_cop_value(self, plant_config):
        with pytest.raises(ValueError, match="cop"):
            HeatPumpPerformanceModel(
                plant_config=plant_config,
                tech_config=_make_perf_config(cop=None),
            ).setup()


@pytest.mark.unit
class TestHeatPumpCostModel:
    def _build(self, plant_config, tech_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "hp_cost",
            HeatPumpCostModel(plant_config=plant_config, tech_config=tech_config),
            promotes=["*"],
        )
        prob.setup()
        return prob

    def test_capex_and_opex(self, plant_config):
        prob = self._build(plant_config, _make_cost_config())
        # Deliver 2 MW for 24 h -> 48 MWh
        prob.set_val("heat_out", np.full(24, 2.0), units="MW")
        prob.run_model()

        # capex = 500_000 * 2 = 1_000_000; fixed_om = 15_000 * 2 = 30_000;
        # variable_om = 2 * 48 = 96; opex = 30_000 + 96 = 30_096
        assert prob.get_val("CapEx", units="USD")[0] == approx(1_000_000.0)
        assert prob.get_val("OpEx", units="USD/year")[0] == approx(30_096.0)
