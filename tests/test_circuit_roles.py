"""Circuit-string parsing: element roles used by the ECM seeds and the tuner."""
import pytest

from battery_oracle._circuit import circuit_roles


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
