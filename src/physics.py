import numpy as np

G: float = 6.67430e-11
C: float = 299_792_458    

def gravitational_time_dilation(mass_kg: np.ndarray, radius_m: np.ndarray) -> np.ndarray:
    """
    Calculate gravitational time dilation using general relativity.

    T_observed / T_proper = sqrt(1 - 2GM / (rc^2))

    Parameters:
        mass_kg: Mass in kilograms
        radius_m: Radius from the mass center in meters

    Returns:
        Time dilation factor (dimensionless), values in [0, 1]
    """
    assert np.all(radius_m > 0), "Radius must be greater than zero."
    
    factor = 1 - (2 * G * mass_kg) / (radius_m * C**2)
    safe_factor = np.clip(factor, 0, 1)  # Clamp negatives due to numerical precision
    return np.sqrt(safe_factor)


def velocity_time_dilation(velocity_m_s: np.ndarray) -> np.ndarray:
    """
    Calculate velocity-based time dilation using special relativity.

    T_observed / T_proper = sqrt(1 - v^2 / c^2)

    Parameters:
        velocity_m_s: Velocity in meters per second

    Returns:
        Time dilation factor (dimensionless), values in [0, 1]
    """
    assert np.all((velocity_m_s >= 0) & (velocity_m_s < C)), "Velocity must be in range [0, c)."

    factor = 1 - (velocity_m_s**2 / C**2)
    return np.sqrt(np.clip(factor, 0, 1))


def combined_time_dilation(gravity_dilation: np.ndarray, velocity_dilation: np.ndarray) -> np.ndarray:
    """
    Combine gravitational and velocity dilation.

    Combined factor = gravitational_dilation * velocity_dilation

    Parameters:
        gravity_dilation: Output from gravitational_time_dilation()
        velocity_dilation: Output from velocity_time_dilation()

    Returns:
        Combined dilation factor
    """
    assert gravity_dilation.shape == velocity_dilation.shape, "Inputs must have the same shape."
    return gravity_dilation * velocity_dilation


def time_difference(t_far: np.ndarray, t_near: np.ndarray) -> np.ndarray:
    """
    Compute time difference (absolute) between two time dilation values.

    Parameters:
        t_far: Time factor at far distance
        t_near: Time factor near gravity or speed source

    Returns:
        Absolute time difference
    """
    return np.abs(t_far - t_near)