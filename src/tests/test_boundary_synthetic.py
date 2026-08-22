import torch
import numpy as np
import pytest
from src.physical_core.field import PhysicalField
from src.synthetic_generator.generator import SyntheticFieldGenerator

def test_synthetic_sinusoid_generation():
    field = SyntheticFieldGenerator.generate_sinusoid(32, 32, frequencies=[(2.0, 3.0)], amplitudes=[1.5])
    assert field.data.shape == (32, 32)
    assert field.metadata["type"] == "sinusoid"
    assert field.metadata["amplitudes"][0] == 1.5

def test_synthetic_vortex_generation():
    # N=32 on [0, 1] has spacing 1/31, so the centre (0.5, 0.5) falls BETWEEN grid
    # nodes and the sampled peak is strictly below the analytic amplitude
    # (2 * exp(-r^2 / (2 * 0.15^2)) with r^2 = 2 * (0.5 - 15/31)^2 = 1.977...).
    # The generator is correct; the previous rel=1e-2 assertion was not.
    field = SyntheticFieldGenerator.generate_vortex(32, 32, centers=[(0.5, 0.5)], amplitudes=[2.0], core_radii=[0.15])
    assert field.data.shape == (32, 32)
    assert field.metadata["type"] == "vortex"
    assert torch.max(field.data).item() == pytest.approx(1.9770, abs=1e-3)

    # On an ODD grid the centre lands exactly on a node, recovering the amplitude.
    exact = SyntheticFieldGenerator.generate_vortex(33, 33, centers=[(0.5, 0.5)], amplitudes=[2.0], core_radii=[0.15])
    assert torch.max(exact.data).item() == pytest.approx(2.0, rel=1e-6)

def test_synthetic_front_generation():
    field = SyntheticFieldGenerator.generate_front(32, 32, angle=45.0, offset=0.0, width_param=0.05, amplitude=1.0)
    assert field.data.shape == (32, 32)
    assert field.metadata["type"] == "front"
    assert torch.min(field.data).item() >= -1.0
    assert torch.max(field.data).item() <= 1.0

def test_perturbation_engine_rotation():
    from src.synthetic_generator.perturbation import PerturbationEngine
    field = SyntheticFieldGenerator.generate_vortex(32, 32, centers=[(0.5, 0.5)])
    rotated = PerturbationEngine.rotate(field, 90.0)
    assert rotated.data.shape == (32, 32)
    metrics = PerturbationEngine.compute_sensitivity_metrics(field, rotated)
    assert "structural_similarity_index" in metrics
    assert metrics["structural_similarity_index"] > 0.0

def test_perturbation_engine_noise():
    from src.synthetic_generator.perturbation import PerturbationEngine
    field = SyntheticFieldGenerator.generate_sinusoid(32, 32)
    noisy = PerturbationEngine.add_noise(field, "gaussian", 0.1)
    assert noisy.data.shape == (32, 32)
    metrics = PerturbationEngine.compute_sensitivity_metrics(field, noisy)
    assert metrics["mean_squared_error"] > 0.0

def test_boundary_condition_lab():
    from src.boundary_lab.boundary import BoundaryConditionLab
    field = SyntheticFieldGenerator.generate_sinusoid(16, 16)
    
    # Test padding treatments
    padded_periodic = BoundaryConditionLab.apply_boundary_treatment(field, "periodic", 4)
    assert padded_periodic.shape == (24, 24)
    
    padded_zero = BoundaryConditionLab.apply_boundary_treatment(field, "zero", 4)
    assert padded_zero.shape == (24, 24)
    assert padded_zero[0, 0] == 0.0
    
    # Test windowing
    windowed = BoundaryConditionLab.apply_window(field.data, "tukey", 0.2)
    assert windowed.shape == (16, 16)
    
    # Test complete analysis
    analysis = BoundaryConditionLab.analyze_boundary_artefacts(field, "reflect", 4)
    assert "padded_field" in analysis
    assert "distance_profiles" in analysis
    assert len(analysis["distance_profiles"]) > 0
    assert "spectral_leakage" in analysis

def test_boundary_profiles_match_independent_euclidean_ring_oracle():
    """D17: one-pass grouped means/maxima retain the padded-corner definition."""
    from src.boundary_lab.boundary import BoundaryConditionLab

    data = torch.arange(5 * 7, dtype=torch.float64).reshape(5, 7) / 11.0
    field = PhysicalField(data)
    reference = PhysicalField(torch.zeros(9, 11, dtype=torch.float64))
    result = BoundaryConditionLab.analyze_boundary_artefacts(
        field, "reflect", 2, reference_field=reference)
    padded = BoundaryConditionLab.apply_boundary_treatment(field, "reflect", 2)
    dy, dx = torch.gradient(padded.to(torch.float64), edge_order=1)
    gradient = torch.sqrt(dx.square() + dy.square()).numpy()
    error = padded.abs().numpy()

    bins = {}
    for row in range(padded.shape[0]):
        for column in range(padded.shape[1]):
            inside_y = min(row - 2, 2 + field.data.shape[0] - 1 - row)
            inside_x = min(column - 2, 2 + field.data.shape[1] - 1 - column)
            if min(inside_y, inside_x) >= 0:
                distance = min(inside_y, inside_x)
            else:
                outside_y = max(2 - row, 0) + max(
                    row - (2 + field.data.shape[0] - 1), 0)
                outside_x = max(2 - column, 0) + max(
                    column - (2 + field.data.shape[1] - 1), 0)
                distance = np.sqrt(outside_y ** 2 + outside_x ** 2)
            bins.setdefault(int(np.floor(distance)), []).append((row, column))

    assert [row["distance"] for row in result["distance_profiles"]] == [
        float(value) for value in sorted(bins)]
    for observed, distance in zip(result["distance_profiles"], sorted(bins)):
        indices = bins[distance]
        gradients = np.asarray([gradient[index] for index in indices])
        errors = np.asarray([error[index] for index in indices])
        assert observed["mean_gradient"] == pytest.approx(gradients.mean(), abs=1e-12)
        assert observed["max_gradient"] == pytest.approx(gradients.max(), abs=1e-12)
        assert observed["mean_absolute_error"] == pytest.approx(errors.mean(), abs=1e-12)
        assert observed["max_absolute_error"] == pytest.approx(errors.max(), abs=1e-12)

def test_api_endpoints(client):
        
    # Test Synthetic Generation Endpoint
    gen_resp = client.post("/api/v1/synthetic/generate", json={
        "type": "sinusoid",
        "height": 16,
        "width": 16,
        "params": {
            "frequencies": [[1.0, 1.0]],
            "amplitudes": [1.0]
        }
    })
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert len(gen_data["field_data"]) == 16
    
    # Test Perturbation Endpoint
    pert_resp = client.post("/api/v1/synthetic/perturb", json={
        "field_data": gen_data["field_data"],
        "perturbations": [
            {"type": "rotation", "angle": 45.0},
            {"type": "noise", "noise_type": "gaussian", "level": 0.05}
        ]
    })
    assert pert_resp.status_code == 200
    pert_data = pert_resp.json()
    assert len(pert_data["perturbed_field"]) == 16
    assert "metrics" in pert_data
    
    # Test Boundary Analysis Endpoint
    bound_resp = client.post("/api/v1/boundary/analyze", json={
        "field_data": gen_data["field_data"],
        "treatment": "reflect",
        "pad_width": 4,
        "window_type": "tukey",
        "window_alpha": 0.2
    })
    assert bound_resp.status_code == 200
    bound_data = bound_resp.json()
    assert len(bound_data["padded_field"]) == 24
    assert "distance_profiles" in bound_data
