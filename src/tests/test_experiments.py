import pytest
import time
import datetime
from src.database.models import Experiment, ExperimentRun, LineageNode, LineageEdge
from src.experiment_engine.engine import DeclarativeExperimentEngine

# db_session / session_factory / client fixtures come from src/tests/conftest.py,
# which uses StaticPool so the in-memory database is shared with the TestClient thread (D22).

def test_parameter_matrix_expansion():
    matrix = {
        "transform_type": ["fft", "dct"],
        "grid_size": [16, 32]
    }
    expanded = DeclarativeExperimentEngine.expand_parameter_matrix(matrix)
    assert len(expanded) == 4
    assert {"transform_type": "fft", "grid_size": 16} in expanded
    assert {"transform_type": "fft", "grid_size": 32} in expanded
    assert {"transform_type": "dct", "grid_size": 16} in expanded
    assert {"transform_type": "dct", "grid_size": 32} in expanded

def test_experiment_engine_execution(db_session, session_factory):
    experiment_config = {
        "name": "Test Experiment",
        "description": "A simple test experiment",
        "parameter_matrix": {
            "freq": [1.0, 2.0]
        },
        "pipeline": [
            {
                "name": "generate_field",
                "action": "generate_synthetic",
                "args": {
                    "type": "sinusoid",
                    "height": 16,
                    "width": 16,
                    "params": {
                        "frequencies": [["{freq}", "{freq}"]]
                    }
                }
            }
        ],
        "metadata": {
            "code_revision": "test-rev-1",
            "dataset_version": "test-ds-1"
        }
    }

    import uuid
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        id=exp_id,
        name="Test Experiment",
        description="A simple test experiment",
        status="PENDING",
        config=experiment_config,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc)
    )
    db_session.add(experiment)
    db_session.commit()

    # D3: without an injected factory the engine would open the file-backed
    # SessionLocal, never see this experiment, and silently leave it PENDING.
    DeclarativeExperimentEngine.execute_experiment(exp_id, session_factory=session_factory)

    db_session.refresh(experiment)
    assert experiment.status == "COMPLETED"

    runs = db_session.query(ExperimentRun).filter(ExperimentRun.experiment_id == exp_id).all()
    assert len(runs) == 2
    for r in runs:
        assert r.status == "COMPLETED"
        
    nodes = db_session.query(LineageNode).filter(LineageNode.experiment_id == exp_id).all()
    assert len(nodes) == 3
    
    edges = db_session.query(LineageEdge).all()
    assert len(edges) == 2

def test_experiment_api_endpoints(client):
    experiment_payload = {
        "name": "API Test Experiment",
        "description": "Testing the experiment pipeline via API",
        "parameter_matrix": {
            "transform_type": ["fft", "dct"]
        },
        "pipeline": [
            {
                "name": "gen",
                "action": "generate_synthetic",
                "args": {
                    "type": "vortex",
                    "height": 16,
                    "width": 16
                }
            },
            {
                "name": "trans",
                "action": "apply_transform",
                "args": {
                    "field_data": "{gen.field_data}",
                    "transform_type": "{transform_type}"
                }
            }
        ],
        "metadata": {
            "code_revision": "api-rev-1",
            "dataset_version": "api-ds-1"
        }
    }

    resp = client.post("/api/v1/experiments", json=experiment_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "API Test Experiment"
    assert data["status"] == "PENDING"
    exp_id = data["id"]

    time.sleep(1.0)

    resp = client.get(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 200
    detail_data = resp.json()
    assert detail_data["status"] in ["COMPLETED", "RUNNING", "PENDING"]
    assert len(detail_data["runs"]) == 2

    resp = client.get(f"/api/v1/experiments/{exp_id}/lineage")
    assert resp.status_code == 200
    lineage_data = resp.json()
    assert "nodes" in lineage_data
    assert "edges" in lineage_data
    assert len(lineage_data["nodes"]) > 0
