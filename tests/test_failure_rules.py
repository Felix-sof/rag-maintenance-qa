"""The failure rules in knowledge/failure_modes.md, checked against all 10,000 rows of the AI4I dataset."""

import pytest

from failure_rules import DATA_RANGE, OSF_LIMITS, TWF_RATE_BY_WEAR, Process, assess, load_dataset, process_of, twf_rate


@pytest.fixture(scope="module")
def records():
    return load_dataset()


def test_the_dataset_is_the_published_one(records):
    assert len(records) == 10_000
    assert sum(r["machine_failure"] for r in records) == 339


@pytest.mark.parametrize("mode", ["HDF", "PWF", "OSF"])
def test_the_documented_rules_reproduce_every_label(records, mode):
    """The knowledge base is right about the data: no missed label and no false alarm, on every row."""
    wrong = [r["udi"] for r in records if (mode in assess(process_of(r))["triggered"]) != bool(r[mode.lower()])]
    assert wrong == []


def test_twf_rates_match_the_dataset(records):
    lower = -1
    for upper, rate in TWF_RATE_BY_WEAR:
        band = [r for r in records if lower < r["tool_wear_min"] <= upper]
        assert sum(r["twf"] for r in band) / len(band) == pytest.approx(rate, abs=1e-4), upper
        lower = upper


def test_data_ranges_match_the_dataset(records):
    for name, (low, high) in DATA_RANGE.items():
        values = [r[name] for r in records]
        assert (min(values), max(values)) == (low, high), name


def process(**overrides):
    values = dict(product_type="L", air_temperature_k=300.0, process_temperature_k=310.0,
                  rotational_speed_rpm=1500, torque_nm=40.0, tool_wear_min=100)
    return Process(**{**values, **overrides})


class TestAssess:
    def test_a_healthy_process(self):
        result = assess(process())
        assert result["failure_expected"] is False and result["triggered"] == []
        assert result["derived"]["power_w"] == 6283  # 40 Nm x 1500 rpm x 2π / 60
        assert result["outside_data_range"] == []

    def test_hdf_needs_both_conditions_and_reports_the_margins(self):
        one = assess(process(process_temperature_k=308.4, rotational_speed_rpm=1400))
        assert one["modes"]["HDF"]["triggered"] is False
        diff, speed = one["modes"]["HDF"]["conditions"]
        assert (diff["met"], diff["value"], diff["margin"]) == (True, 8.4, 0.2)
        assert (speed["met"], speed["margin"]) == (False, 20)

        both = assess(process(process_temperature_k=308.4, rotational_speed_rpm=1300, torque_nm=45))
        assert both["triggered"] == ["HDF"]

    def test_pwf_fires_on_too_little_or_too_much_power(self):
        assert assess(process(torque_nm=20, rotational_speed_rpm=1500))["triggered"] == ["PWF"]  # 3,142 W
        assert assess(process(torque_nm=60, rotational_speed_rpm=1500))["triggered"] == ["PWF"]  # 9,425 W

    @pytest.mark.parametrize("product_type", ["L", "M", "H"])
    def test_osf_limit_depends_on_the_product_type(self, product_type):
        limit = OSF_LIMITS[product_type]
        just_under = assess(process(product_type=product_type, tool_wear_min=200, torque_nm=limit / 200 - 0.5))
        just_over = assess(process(product_type=product_type, tool_wear_min=200, torque_nm=limit / 200 + 0.5))
        assert "OSF" not in just_under["triggered"] and "OSF" in just_over["triggered"]

    def test_twf_is_a_rate_not_a_verdict(self):
        twf = assess(process(tool_wear_min=215))["modes"]["TWF"]
        assert twf["triggered"] is None and twf["reached_window"] is True
        assert twf["dataset_rate"] == twf_rate(215) == 0.0489

    def test_values_no_record_resembles_are_flagged(self):
        assert assess(process(air_temperature_k=290.0))["outside_data_range"] == ["air_temperature_k"]

    @pytest.mark.parametrize("overrides", [{"product_type": "X"}, {"torque_nm": -1}, {"tool_wear_min": float("nan")}])
    def test_invalid_input_is_rejected(self, overrides):
        with pytest.raises(ValueError):
            process(**overrides)
