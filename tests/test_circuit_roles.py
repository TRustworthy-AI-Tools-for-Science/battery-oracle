"""Circuit-string parsing: element roles used by the ECM seeds and the tuner."""
import pytest

from battery_oracle.circuit import circuit_roles


@pytest.mark.parametrize("circuit,ohmic,pairs,series", [
    ("R1-[R2,P3]-[R4,P5]", "R1", [("R2", "P3"), ("R4", "P5")], []),
    ("R1-P2-[R3,P4]-[R5,P6]", "R1", [("R3", "P4"), ("R5", "P6")], ["P2"]),
    ("[R1,P2]-R3", "R3", [("R1", "P2")], []),
    ("R7", "R7", [], []),
])
def test_roles_partition_the_circuit(circuit, ohmic, pairs, series):
    roles = circuit_roles(circuit)
    assert roles.ohmic == ohmic
    assert roles.pairs == pairs
    assert roles.series_cpes == series
    # every label belongs to exactly one role
    covered = [roles.ohmic] + [x for r, p in roles.pairs for x in (r, f"{p}w", f"{p}n")]
    covered += [x for p in roles.series_cpes for x in (f"{p}w", f"{p}n")]
    assert sorted(covered) == sorted(roles.labels)


def test_circuit_without_series_resistor_is_rejected():
    with pytest.raises(ValueError, match="ohmic"):
        circuit_roles("[R1,P2]-[R3,P4]")


def test_ecm_std_follows_the_label_order_of_the_means():
    """The posterior std vector is in circuit-label order like the means, whatever order
    AutoEIS lists its variables in."""
    import numpy as np

    from battery_oracle import PyBaMMOracle

    o = PyBaMMOracle(circuit="R1-[R2,P3]")
    labels = list(o._ecm_param_names)
    samples = {k: np.array([0.0, 2.0 * (i + 1)]) for i, k in enumerate(labels)}
    std = o._ecm_std_half({"_raw_samples": samples, "_raw_variables": labels[::-1]})
    assert np.allclose(std, [float(np.std(samples[k])) for k in labels])


def test_flip_fix_needs_a_circuit():
    from battery_oracle.circuit import fix_parameter_flips_dicts

    with pytest.raises(ValueError, match="circuit"):
        fix_parameter_flips_dicts([{}], None)
