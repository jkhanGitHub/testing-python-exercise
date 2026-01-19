"""
Integration tests for diffusion2d.py
"""
import pytest
import numpy as np
from diffusion2d import SolveDiffusion2D

def test_initialize_physical_parameters_integration():
    """
    Test interaction between initialize_domain and initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    w, h, dx, dy = 10., 10., 0.1, 0.1
    d, T_cold, T_hot = 4., 300., 700.
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)
    
    # Expected dt calculation
    dx2, dy2 = dx * dx, dy * dy
    expected_dt = dx2 * dy2 / (2 * d * (dx2 + dy2))
    
    assert abs(solver.dt - expected_dt) < 1e-7, "dt is not calculated correctly in integration"

def test_set_initial_condition_integration():
    """
    Test interaction involving set_initial_condition
    """
    solver = SolveDiffusion2D()
    w, h, dx, dy = 10., 10., 0.1, 0.1
    d, T_cold, T_hot = 4., 300., 700.
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)
    
    u = solver.set_initial_condition()
    
    # Manual calculation of expected field
    nx = int(w / dx)
    ny = int(h / dy)
    expected_u = T_cold * np.ones((nx, ny))
    
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = T_hot
                
    assert u.shape == expected_u.shape
    np.testing.assert_allclose(u, expected_u, err_msg="Initial condition u does not match expected values")
