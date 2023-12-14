import pytest
from mlcraft.check_dimensions import assert_dimensions_match
from mlcraft.errors import Error


def test_correct_simple():
    layers = [
        {"id": 0, "type": "Target", "parameters": {"width": 1}, "parents": []},
        {"id": 1, "type": "Data", "parameters": {"width": 2}, "parents": []},
        {
            "id": 2,
            "type": "Linear",
            "parameters": {"inFeatures": 2, "outFeatures": 1},
            "parents": [1],
        },
        {"id": 3, "type": "ReLU", "parameters": {}, "parents": [2]},
        {"id": 4, "type": "MSELoss", "parameters": {}, "parents": [0, 3]},
        {"id": 5, "type": "Output", "parameters": {}, "parents": [3]},
    ]
    # Assert no exception is raised
    assert_dimensions_match(layers)


def test_correct_harder():
    layers = [
        {"id": 0, "type": "Target", "parameters": {"width": 1}, "parents": []},
        {"id": 1, "type": "Data", "parameters": {"width": 2}, "parents": []},
        {
            "id": 2,
            "type": "Linear",
            "parameters": {"inFeatures": 2, "outFeatures": 10},
            "parents": [1],
        },
        {"id": 3, "type": "ReLU", "parameters": {}, "parents": [2]},
        {
            "id": 4,
            "type": "Linear",
            "parameters": {"inFeatures": 10, "outFeatures": 10},
            "parents": [3],
        },
        {"id": 5, "type": "ReLU", "parameters": {}, "parents": [4]},
        {"id": 6, "type": "Sum", "parameters": {}, "parents": [3, 5]},
        {
            "id": 7,
            "type": "Linear",
            "parameters": {"inFeatures": 10, "outFeatures": 1},
            "parents": [6],
        },
        {"id": 8, "type": "ReLU", "parameters": {}, "parents": [7]},
        {"id": 9, "type": "MSELoss", "parameters": {}, "parents": [0, 8]},
        {"id": 10, "type": "Output", "parameters": {}, "parents": [8]},
    ]
    assert_dimensions_match(layers)


def test_mismatch_simple():
    layers = [
        {"id": 0, "type": "Target", "parameters": {"width": 1}, "parents": []},
        {"id": 1, "type": "Data", "parameters": {"width": 2}, "parents": []},
        {
            "id": 2,
            "type": "Linear",
            "parameters": {"inFeatures": 2, "outFeatures": 2},
            "parents": [1],
        },
        {"id": 3, "type": "ReLU", "parameters": {}, "parents": [2]},
        {"id": 4, "type": "MSELoss", "parameters": {}, "parents": [0, 3]},
        {"id": 5, "type": "Output", "parameters": {}, "parents": [3]},
    ]

    with pytest.raises(Error) as e:
        assert_dimensions_match(layers)

    assert e.match(r".*4.*match.*dimensions")
    assert e.value.payload["problemNode"] == 4


def test_mismatch_harder():
    layers = [
        {"id": 0, "type": "Target", "parameters": {"width": 1}, "parents": []},
        {"id": 1, "type": "Data", "parameters": {"width": 2}, "parents": []},
        {
            "id": 2,
            "type": "Linear",
            "parameters": {"inFeatures": 2, "outFeatures": 10},
            "parents": [1],
        },
        {"id": 3, "type": "ReLU", "parameters": {}, "parents": [2]},
        {
            "id": 4,
            "type": "Linear",
            "parameters": {"inFeatures": 10, "outFeatures": 18},
            "parents": [3],
        },
        {"id": 5, "type": "ReLU", "parameters": {}, "parents": [4]},
        {"id": 6, "type": "Sum", "parameters": {}, "parents": [3, 5]},
        {
            "id": 7,
            "type": "Linear",
            "parameters": {"inFeatures": 10, "outFeatures": 1},
            "parents": [6],
        },
        {"id": 8, "type": "ReLU", "parameters": {}, "parents": [7]},
        {"id": 9, "type": "MSELoss", "parameters": {}, "parents": [0, 8]},
        {"id": 10, "type": "Output", "parameters": {}, "parents": [8]},
    ]

    with pytest.raises(Error) as e:
        assert_dimensions_match(layers)

    assert e.match(r".*6.*match.*dimensions")
    assert e.value.payload["problemNode"] == 6
