import pytest
import datetime
import uuid
from src.database.models import Experiment, ExperimentRun, Hypothesis
from src.hypothesis_engine.engine import PatternDiscoveryEngine

# db_session / session_factory / client fixtures come from src/tests/conftest.py,
# which uses StaticPool so the in-memory database is shared with the TestClient thread (D22).

def test_correlation_pattern_discovery(db_session):
    # Create a test experiment
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        id=exp_id,
        name="Test Correlation Experiment",
        description="Testing correlation pattern discovery",
        status="COMPLETED",
        config={
            "parameter_matrix": {
                "freq": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            "pipeline": [
                {
                    "name": "gen",
                    "action": "generate_synthetic",
                    "args": {"type": "sinusoid"}
                }
            ]
        }
    )
    db_session.add(experiment)
    db_session.commit()

    # Create completed runs with a clear positive correlation between freq and mean_squared_error
    for freq in [1.0, 2.0, 3.0, 4.0, 5.0]:
        run = ExperimentRun(
            id=str(uuid.uuid4()),
            experiment_id=exp_id,
            parameters={"freq": freq},
            status="COMPLETED",
            results={
                "trans_metrics": {
                    "mean_squared_error": freq * 0.1
                }
            }
        )
        db_session.add(run)
    db_session.commit()

    # Run discovery
    hypotheses = PatternDiscoveryEngine.discover_patterns(db_session, experiment_ids=[exp_id])
    assert len(hypotheses) == 1
    h = hypotheses[0]
    assert h.pattern_type == "correlation"
    assert h.confidence == pytest.approx(1.0)
    assert "freq" in h.parameters_analyzed
    assert "trans_metrics.mean_squared_error" in h.metrics_analyzed
    
    # Since it's an error metric and correlation is positive, we want smaller values of freq
    proposed_config = h.proposed_experiment_config
    assert proposed_config is not None
    assert "freq" in proposed_config["parameter_matrix"]
    assert proposed_config["parameter_matrix"]["freq"][0] < 1.0

def test_categorical_pattern_discovery(db_session):
    # Create a test experiment
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        id=exp_id,
        name="Test Categorical Experiment",
        description="Testing categorical pattern discovery",
        status="COMPLETED",
        config={
            "parameter_matrix": {
                "transform_type": ["fft", "dct"]
            },
            "pipeline": [
                {
                    "name": "trans",
                    "action": "apply_transform",
                    "args": {}
                }
            ]
        }
    )
    db_session.add(experiment)
    db_session.commit()

    # Create completed runs where dct performs significantly better (lower error) than fft
    runs_data = [
        ("fft", 0.5),
        ("fft", 0.6),
        ("dct", 0.1),
        ("dct", 0.12)
    ]
    for transform_type, error in runs_data:
        run = ExperimentRun(
            id=str(uuid.uuid4()),
            experiment_id=exp_id,
            parameters={"transform_type": transform_type},
            status="COMPLETED",
            results={
                "trans_metrics": {
                    "mean_squared_error": error
                }
            }
        )
        db_session.add(run)
    db_session.commit()

    # Run discovery
    hypotheses = PatternDiscoveryEngine.discover_patterns(db_session, experiment_ids=[exp_id])
    assert len(hypotheses) == 1
    h = hypotheses[0]
    assert h.pattern_type == "categorical_opt"
    assert "transform_type" in h.parameters_analyzed
    assert "dct" in h.description
    
    # Proposed config should fix transform_type to ["dct"]
    proposed_config = h.proposed_experiment_config
    assert proposed_config is not None
    assert proposed_config["parameter_matrix"]["transform_type"] == ["dct"]

def test_hypothesis_api_endpoints(client, db_session):
    # Create a test experiment and runs
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        id=exp_id,
        name="API Test Experiment",
        description="Testing API endpoints",
        status="COMPLETED",
        config={
            "parameter_matrix": {
                "freq": [1.0, 2.0, 3.0]
            },
            "pipeline": []
        }
    )
    db_session.add(experiment)
    
    for freq in [1.0, 2.0, 3.0]:
        run = ExperimentRun(
            id=str(uuid.uuid4()),
            experiment_id=exp_id,
            parameters={"freq": freq},
            status="COMPLETED",
            results={
                "metrics": {
                    "mean_squared_error": freq * 0.1
                }
            }
        )
        db_session.add(run)
    db_session.commit()

    # Test Discover Endpoint
    resp = client.post("/api/v1/hypothesis/discover", json={
        "experiment_ids": [exp_id],
        "confidence_threshold": 0.5
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["pattern_type"] == "correlation"
    assert data[0]["confidence"] == pytest.approx(1.0)

    # Test Proposals Endpoint
    resp = client.get("/api/v1/hypothesis/proposals")
    assert resp.status_code == 200
    proposals = resp.json()
    assert len(proposals) == 1
    assert proposals[0]["pattern_type"] == "correlation"
