import datetime
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import relationship
from src.database.session import Base

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="PENDING")
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))

    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")
    lineage_nodes = relationship("LineageNode", back_populates="experiment", cascade="all, delete-orphan")

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(String, default="PENDING")
    results = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    experiment = relationship("Experiment", back_populates="runs")
    lineage_nodes = relationship("LineageNode", back_populates="run", cascade="all, delete-orphan")

class LineageNode(Base):
    __tablename__ = "lineage_nodes"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String, ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # "dataset", "code_revision", "field", "coefficients", "metrics"
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    experiment = relationship("Experiment", back_populates="lineage_nodes")
    run = relationship("ExperimentRun", back_populates="lineage_nodes")

class LineageEdge(Base):
    __tablename__ = "lineage_edges"

    id = Column(String, primary_key=True, index=True)
    source_id = Column(String, ForeignKey("lineage_nodes.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(String, ForeignKey("lineage_nodes.id", ondelete="CASCADE"), nullable=False)
    relation = Column(String, nullable=False)  # "executed_by", "input_to", "sliced_from"

class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id = Column(String, primary_key=True, index=True)
    experiment_ids = Column(JSON, nullable=False)  # List of experiment IDs analyzed
    pattern_type = Column(String, nullable=False)  # "correlation" or "categorical_opt"
    description = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    metrics_analyzed = Column(JSON, nullable=False)
    parameters_analyzed = Column(JSON, nullable=False)
    proposed_experiment_config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
