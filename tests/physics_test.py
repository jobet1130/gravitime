import numpy as np
import pytest
from src.physics import (
    gravitational_time_dilation,
    velocity_time_dilation,
    combined_time_dilation,
    time_difference,
    C
)

def test_gravitational_time_dilation_clip():
    mass = np.array([1e50])
    radius = np.array([1e3])
    result = gravitational_time_dilation(mass, radius)
    assert np.all(result <= 1) and np.all(result >= 0), "Result must be clipped to [0, 1]"

def test_gravitational_time_dilation_zero_radius():
    mass = np.array([5.972e24])
    radius = np.array([0.0])
    with pytest.raises(AssertionError):
        gravitational_time_dilation(mass, radius)

def test_gravitational_time_dilation_valid():
    mass = np.array([5.972e24])
    radius = np.array([6.371e6])
    result = gravitational_time_dilation(mass, radius)
    assert 0 < result[0] < 1, "Gravitational dilation should be between 0 and 1."

def test_velocity_time_dilation_valid():
    v = np.array([100_000.0])
    result = velocity_time_dilation(v)
    assert 0 < result[0] < 1, "Velocity dilation should be between 0 and 1."

def test_velocity_time_dilation_invalid_velocity():
    v = np.array([C * 1.1])
    with pytest.raises(AssertionError):
        velocity_time_dilation(v)

def test_combined_time_dilation_shapes():
    g = np.array([0.9, 0.8])
    v = np.array([0.99, 0.95])
    result = combined_time_dilation(g, v)
    expected = g * v
    np.testing.assert_array_almost_equal(result, expected)

def test_combined_time_dilation_shape_mismatch():
    g = np.array([0.9])
    v = np.array([0.99, 0.95])
    with pytest.raises(AssertionError):
        combined_time_dilation(g, v)

def test_time_difference():
    t1 = np.array([0.95])
    t2 = np.array([0.90])
    result = time_difference(t1, t2)
    assert result == pytest.approx(0.05)

def test_time_difference_array():
    t1 = np.array([0.95, 0.99])
    t2 = np.array([0.90, 0.98])
    result = time_difference(t1, t2)
    np.testing.assert_array_almost_equal(result, [0.05, 0.01])

def test_output_type_and_shape():
    mass = np.array([5.972e24, 1.0e25])
    radius = np.array([6.371e6, 7.0e6])
    result = gravitational_time_dilation(mass, radius)
    
    assert isinstance(result, np.ndarray), "Output should be a NumPy array"
    assert result.shape == mass.shape, "Output shape must match input shape"
    assert result.dtype == np.float64, "Output should be of dtype float64"
