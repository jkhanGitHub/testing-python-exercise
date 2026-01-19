"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20., h=40., dx=0.2, dy=0.5)
    assert solver.nx == 100, "nx is not correctly set"
    assert solver.ny == 80, "ny is not correctly set"


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=60., h=60., dx=0.2, dy=0.3)
    solver.initialize_physical_parameters(d=6., T_cold=200., T_hot=400.)
    
    # Expected dt = 0.04 * 0.09 / (2 * 6 * (0.04 + 0.09)) = 0.0036 / 1.56 = 0.00230769
    assert abs(solver.dt - 0.00230769) < 1e-7, "dt is not correctly set"


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=60., h=60., dx=0.2, dy=0.3)
    solver.initialize_physical_parameters(d=6., T_cold=200., T_hot=400.)
    u0 = solver.set_initial_condition()
    assert u0.shape == (300, 200), "Initial condition array has wrong shape"