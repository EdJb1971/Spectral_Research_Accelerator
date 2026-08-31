import { useState, useEffect, useRef } from 'react';
import { Heatmap2D } from './components/Heatmap2D';
import { LineChart } from './components/LineChart';
import { LineageGraph } from './components/LineageGraph';
import { FieldExportBar, TableExportBar } from './components/ExportBar';
import { FigureExport } from './components/FigureExport';
import { FieldImport } from './components/FieldImport';
import { TrainingReadiness } from './components/TrainingReadiness';
import { DTCWTScientificView } from './components/DTCWTScientificView';
import { EvaluationEvidence } from './components/EvaluationEvidence';
import FindingsView from './components/FindingsView';
import AcquisitionView from './components/AcquisitionView';
import DomainAnalysisView from './components/DomainAnalysisView';
import PreregistrationView from './components/PreregistrationView';
import EvidenceView from './components/EvidenceView';
import StructureMiningView from './components/StructureMiningView';
import CrossDomainRecordView from './components/CrossDomainRecordView';
import ReviewView from './components/ReviewView';
import DatasetCapabilityProfile from './components/DatasetCapabilityProfile';
import ExperimentComposer from './components/ExperimentComposer';
import { ExperimentReceiptPanel } from './components/ExperimentReceipt';
import { ExperimentQualificationPanel } from './components/ExperimentQualification';
import { apiService } from './services/api';
import * as types from './types/api';
import {
  BookOpen,
  Layers,
  Wind,
  Sliders,
  Activity,
  BarChart2,
  FileCode,
  Lightbulb,
  Globe,
  Database,
  Play,
  RotateCcw,
  Plus,
  Trash2,
  CheckCircle,
  XCircle,
  Loader2,
  TrendingUp,
  SlidersHorizontal,
  Server,
  Code,
  ShieldCheck,
  AlertTriangle,
  Cloud,
  WifiOff,
  Boxes,
  Waypoints,
  FileCheck2,
  FilePlus2,
  FileLock2,
  Lock,
  MessageSquare
} from 'lucide-react';

const WORKFLOW_NAV = [
  { section: 'Acquire', items: [{ id: 'acquire', name: 'Acquire data', icon: Cloud }] },
  {
    section: 'Analyse', items: [
      { id: 'domainWorkbench', name: 'Cross-domain analysis', icon: Globe, operation: 'cross_domain_analysis' },
      { id: 'preregistration', name: 'Preregistration', icon: Lock },
      { id: 'synthetic', name: 'Synthetic generator', icon: Layers, context: 'Gridded field line' },
      { id: 'meteorological', name: 'Meteorological data', icon: Wind, context: 'Gridded field line' },
      { id: 'boundary', name: 'Boundary-condition lab', icon: Sliders, context: 'Gridded field line', operation: 'boundary_lab' },
      { id: 'spectral', name: 'Spectral transforms', icon: Activity, context: 'Gridded field line', operation: 'dtcwt_spatial' },
      { id: 'analysis', name: 'Diagnostics', icon: BarChart2, context: 'Gridded field line', operation: 'gridded_diagnostics' },
      { id: 'mining', name: 'Structure mining', icon: Boxes, operation: 'structure_mining' },
      { id: 'crossDomainRecord', name: 'Cross-domain record', icon: Waypoints, operation: 'cross_domain_analysis' },
      { id: 'hypothesis', name: 'Automated hypotheses', icon: Lightbulb },
    ],
  },
  {
    section: 'Evidence', items: [
      { id: 'evidence', name: 'Evidence record', icon: FilePlus2 },
      { id: 'experimentComposer', name: 'Experiment Composer', icon: FileLock2 },
      { id: 'declarative', name: 'Legacy parameter sweeps', icon: FileCode, context: 'Gridded field line' },
      { id: 'evaluation', name: 'Forecast evaluation', icon: FileCheck2, context: 'Gridded field line', operation: 'forecast_evaluation' },
    ],
  },
  { section: 'Review', note: 'Recorded argument; never claim permission', items: [
    { id: 'review', name: 'Recorded review', icon: MessageSquare },
  ] },
  { section: 'Read', items: [{ id: 'findings', name: 'Findings', icon: BookOpen }] },
  { section: 'Platform', items: [{ id: 'platform', name: 'Platform & evidence', icon: ShieldCheck }] },
] as const;

export default function App() {
  const [activeTab, setActiveTab] = useState('acquire');
  const workspaceHeadingRef = useRef<HTMLHeadingElement>(null);
  const hasMountedRef = useRef(false);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  // TG11.0: context belongs to the shell, not to whichever workflow panel is mounted.
  const [selectedRecord, setSelectedRecord] = useState<types.ChannelRecordSelection | null>(null);
  const [selectedCapability, setSelectedCapability] = useState<types.DatasetCapabilityProfile | null>(null);
  const [selectedStudyId, setSelectedStudyId] = useState<string>('');

  // T3.5.22: platform status and the ERA5 crop tools. Every one of these endpoints existed
  // and had no consumer, so the platform could report its device, executor, schema revision,
  // benchmark results and data provenance and a researcher could see none of it.
  const [health, setHealth] = useState<types.HealthResponse | null>(null);
  const [benchmarks, setBenchmarks] = useState<types.BenchmarkResponse[]>([]);
  const [dataSources, setDataSources] = useState<types.DataSourceInfo[]>([]);
  // T3.5.24: evidence a researcher can generate, and capabilities they can discover.
  const [benchmarkRun, setBenchmarkRun] = useState<types.BenchmarkSuiteResponse | null>(null);
  const [benchmarkSeed, setBenchmarkSeed] = useState(20260819);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [registryTransforms, setRegistryTransforms] = useState<types.RegistryEntry[]>([]);
  const [registryActions, setRegistryActions] = useState<types.RegistryEntry[]>([]);
  const [evaluationReports, setEvaluationReports] = useState<types.EvaluationReport[]>([]);
  const [receiptImporting, setReceiptImporting] = useState(false);
  const [importedProvenance, setImportedProvenance] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Core Scientific Fields State (shared or passed between tabs)
  const [primaryField, setPrimaryField] = useState<number[][]>(() =>
    Array.from({ length: 32 }, (_, r) =>
      Array.from({ length: 32 }, (_, c) => Math.sin(r / 4) * Math.cos(c / 4))
    )
  );
  const [primaryCoords, setPrimaryCoords] = useState<Record<string, number[]>>({
    x: Array.from({ length: 32 }, (_, i) => i / 31),
    y: Array.from({ length: 32 }, (_, i) => i / 31),
  });
  const [primaryMetadata, setPrimaryMetadata] = useState<Record<string, any>>({
    type: 'initial_sinusoid',
  });

  // --- TAB 1 STATE: Synthetic Gen & Perturbation ---
  const [genType, setGenType] = useState('sinusoid');
  const [gridSize, setGridSize] = useState(32);
  const [freqX, setFreqX] = useState(2.0);
  const [freqY, setFreqY] = useState(2.0);
  const [sinAmplitude, setSinAmplitude] = useState(1.0);
  const [vortexCenterX, setVortexCenterX] = useState(0.5);
  const [vortexCenterY, setVortexCenterY] = useState(0.5);
  const [vortexAmp, setVortexAmp] = useState(1.5);
  const [vortexRadius, setVortexRadius] = useState(0.15);
  const [frontAngle, setFrontAngle] = useState(45.0);
  const [frontOffset, setFrontOffset] = useState(0.0);
  const [frontWidth, setFrontWidth] = useState(0.08);
  const [frontAmp, setFrontAmp] = useState(1.0);

  // Perturbations builder
  const [perturbations, setPerturbations] = useState<types.PerturbationItem[]>([]);
  const [newPertType, setNewPertType] = useState('rotation');
  const [newPertAngle, setNewPertAngle] = useState(15.0);
  const [newPertShiftX, setNewPertShiftX] = useState(0.1);
  const [newPertShiftY, setNewPertShiftY] = useState(0.1);
  const [newPertNoiseType, setNewPertNoiseType] = useState('gaussian');
  const [newPertNoiseLevel, setNewPertNoiseLevel] = useState(0.1);

  const [perturbedField, setPerturbedField] = useState<number[][] | null>(null);
  const [perturbationMetrics, setPerturbationMetrics] = useState<types.PerturbResponse['metrics'] | null>(null);

  // --- TAB 2 STATE: Meteorological Explorer ---
  const [datasets, setDatasets] = useState<types.DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('era5_reanalysis');
  const [selectedVariable, setSelectedVariable] = useState<string>('t2m');
  const [selectedTime, _setSelectedTime] = useState<string>('');
  const [selectedLevel, setSelectedLevel] = useState<number>(500);
  const [latMin, setLatMin] = useState<number>(-45.0);
  const [latMax, setLatMax] = useState<number>(45.0);
  const [lonMin, setLonMin] = useState<number>(-90.0);
  const [lonMax, setLonMax] = useState<number>(90.0);
  const [slicedField, setSlicedField] = useState<number[][] | null>(null);
  const [slicedCoords, setSlicedCoords] = useState<Record<string, number[]>>({});
  const [_slicedMetadata, setSlicedMetadata] = useState<Record<string, any>>({});

  // --- TAB 3 STATE: Boundary-Condition Lab ---
  const [boundaryTreatment, setBoundaryTreatment] = useState('reflect');
  const [padWidth, setPadWidth] = useState(8);
  const [windowType, setWindowType] = useState<string>('tukey');
  const [windowAlpha, setWindowAlpha] = useState(0.2);
  const [paddedField, setPaddedField] = useState<number[][] | null>(null);
  const [distanceProfiles, setDistanceProfiles] = useState<types.DistanceProfile[]>([]);
  const [spectralLeakage, setSpectralLeakage] = useState<number | null>(null);

  // --- TAB 4 STATE: Spectral Transform Engine ---
  const [transformType, setTransformType] = useState('fft');
  const [waveletLevels, setWaveletLevels] = useState(1);
  const [waveletFamily, setWaveletFamily] = useState('db2');
  const [crossoverFreq, setCrossoverFreq] = useState(0.2);
  const [mixingWeight, setMixingWeight] = useState(0.5);
  const [reconstructedField, setReconstructedField] = useState<number[][] | null>(null);
  const [transformMetrics, setTransformMetrics] = useState<types.TransformResponse['metrics'] | null>(null);
  const [transformCoefficients, setTransformCoefficients] = useState<Record<string, any> | null>(null);
  const [trainingCatalogue, setTrainingCatalogue] = useState<types.TrainingRepresentationCatalogue | null>(null);

  // --- TAB 5 STATE: Analysis & Diagnostics ---
  const [forecastNoise, setForecastNoise] = useState(0.15);
  // An explicit, visible seed. A diagnostic whose input cannot be regenerated is not a
  // measurement of anything, and the seed has to be on screen for the run to be quotable.
  const [forecastSeed, setForecastSeed] = useState(20260820);
  const [forecastProvenance, setForecastProvenance] = useState<any | null>(null);
  const [diagnosticsResults, setDiagnosticsResults] = useState<types.DiagnosticsResponse | null>(null);
  const [scaleDecompResults, setScaleDecompResults] = useState<types.ErrorDecompositionResponse['scale_decomposition'] | null>(null);
  const [boundaryDecompResults, setBoundaryDecompResults] = useState<types.ErrorDecompositionResponse['boundary_decomposition'] | null>(null);

  // --- TAB 6 STATE: Experiment Engine ---
  const [experimentJson, setExperimentJson] = useState<string>(() => {
    return JSON.stringify({
      name: "Spectral Accuracy Parameter Sweep",
      description: "Cartesian parameter sweep measuring FFT vs DCT error on sinusoids.",
      parameter_matrix: {
        freq: [1.0, 2.0, 3.0],
        transform_type: ["fft", "dct"]
      },
      pipeline: [
        {
          name: "gen",
          action: "generate_synthetic",
          args: {
            type: "sinusoid",
            height: 32,
            width: 32,
            params: {
              frequencies: [["{freq}", "{freq}"]]
            }
          }
        },
        {
          name: "trans",
          action: "apply_transform",
          args: {
            field_data: "{gen.field_data}",
            transform_type: "{transform_type}"
          }
        }
      ],
      metadata: {
        code_revision: "v1.2.0-beta",
        dataset_version: "syn-01"
      }
    }, null, 2);
  });
  const [_submittedExperimentId, setSubmittedExperimentId] = useState<string | null>(null);
  const [experimentDetail, setExperimentDetail] = useState<types.ExperimentDetailResponse | null>(null);
  const [lineageData, setLineageData] = useState<types.LineageResponse | null>(null);

  // --- TAB 7 STATE: Hypothesis & Proposals ---
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
  const [hypotheses, setHypotheses] = useState<types.HypothesisResponse[]>([]);
  const [_proposals, setProposals] = useState<types.HypothesisResponse[]>([]);

  // Check Backend Connectivity at startup
  useEffect(() => {
    checkConnection();
  }, []);

  useEffect(() => {
    if (backendConnected !== true) return;
    apiService.listTrainingRepresentations(
      waveletLevels,
      waveletFamily,
      primaryField.length,
      primaryField[0]?.length || 1,
    ).then(setTrainingCatalogue).catch((err: any) => {
      setTrainingCatalogue(null);
      setError(`Training capability evidence could not be loaded: ${err.message}`);
    });
  }, [backendConnected, waveletLevels, waveletFamily, primaryField]);

  const checkConnection = async () => {
    try {
      const ds = await apiService.listDatasets();
      setDatasets(ds);
      setBackendConnected(true);
      setError(null);
    } catch (err: any) {
      // The backend is unreachable. Previously this substituted fabricated datasets and every
      // tab fell back to fabricating results in the browser - fields, transforms, diagnostics,
      // even experiment IDs - with nothing on the individual result saying so. A spectral slope
      // computed from `Math.random()` looked exactly like one computed from ERA5. Every one of
      // those paths has been deleted: with no backend there is no data, and the UI says so.
      setBackendConnected(false);
      setDatasets([]);
      setError(
        `Backend unreachable at /api/v1 (${err.message}). Nothing can be computed until it is ` +
        `running - start it with start_platform.ps1. No results are fabricated in its absence.`
      );
    }
  };

  // --- MUTATORS & ACTIONS ---

  // Generate Synthetic Field
  const handleGenerateSynthetic = async () => {
    setLoading(true);
    setError(null);
    try {
      let params: Record<string, any> = {};
      if (genType === 'sinusoid') {
        params = {
          frequencies: [[freqX, freqY]],
          amplitudes: [sinAmplitude],
          phases: [[0.0, 0.0]]
        };
      } else if (genType === 'vortex') {
        params = {
          centers: [[vortexCenterX, vortexCenterY]],
          amplitudes: [vortexAmp],
          core_radii: [vortexRadius]
        };
      } else if (genType === 'front') {
        params = {
          angle: frontAngle,
          offset: frontOffset,
          width_param: frontWidth,
          amplitude: frontAmp
        };
      }
        const res = await apiService.generateSynthetic({
          type: genType,
          height: gridSize,
          width: gridSize,
          params
        });
        setPrimaryField(res.field_data);
        setPrimaryCoords(res.coords);
        setPrimaryMetadata(res.metadata);
      
      setPerturbedField(null);
      setPerturbationMetrics(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };


  // Add Perturbation Step
  const addPerturbation = () => {
    let item: types.PerturbationItem = { type: newPertType };
    if (newPertType === 'rotation') {
      item.angle = newPertAngle;
    } else if (newPertType === 'translation') {
      item.shift_x = newPertShiftX;
      item.shift_y = newPertShiftY;
    } else {
      item.noise_type = newPertNoiseType;
      item.level = newPertNoiseLevel;
    }
    setPerturbations([...perturbations, item]);
  };

  // Apply Perturbations
  const handleApplyPerturbations = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.perturbField({
        field_data: primaryField,
        perturbations
      });
      setPerturbedField(res.perturbed_field);
      setPerturbationMetrics(res.metrics);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };


  // --- TAB 2 ACTIONS: Met Data Slice ---
  const handleSliceDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.sliceDataset({
        dataset_id: selectedDatasetId,
        variable: selectedVariable,
        time: selectedTime || undefined,
        level: selectedLevel,
        lat_range: [latMin, latMax],
        lon_range: [lonMin, lonMax]
      });
      setSlicedField(res.field_data);
      setSlicedCoords(res.coords);
      setSlicedMetadata(res.metadata);
      // Also load into primary field for further processing in transforms/boundary tabs
      setPrimaryField(res.field_data);
      setPrimaryCoords(res.coords);
      setPrimaryMetadata(res.metadata);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- TAB 3 ACTIONS: Boundary Conditions ---
  const handleAnalyzeBoundary = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.analyzeBoundary({
        field_data: primaryField,
        treatment: boundaryTreatment,
        pad_width: padWidth,
        window_type: windowType || undefined,
        window_alpha: windowAlpha
      });
      setPaddedField(res.padded_field);
      setDistanceProfiles(res.distance_profiles);
      setSpectralLeakage(res.spectral_leakage);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- TAB 4 ACTIONS: Spectral Transforms ---
  const handleApplyTransform = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.applyTransform({
        field_data: primaryField,
        transform_type: transformType,
        config: {
          levels: waveletLevels,
          wavelet: waveletFamily,
          crossover_freq: crossoverFreq,
          mixing_weight: mixingWeight
        }
      });
      setReconstructedField(res.reconstructed_field);
      setTransformMetrics(res.metrics);
      setTransformCoefficients(res.coefficients);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- TAB 5 ACTIONS: Diagnostics ---
  const handleComputeDiagnostics = async () => {
    setLoading(true);
    setError(null);
    try {
      // The synthetic "forecast" is now drawn by the backend's seeded perturbation engine.
      // It used to be built here with `Math.random()`: unseeded, so no diagnostic computed
      // from it could ever be reproduced, and *uniform* despite the control being labelled
      // "StDev". The platform's whole seed discipline (T3.5.12) existed and this bypassed it.
      const perturbed = await apiService.perturbField({
        field_data: primaryField,
        perturbations: [{ type: 'noise', noise_type: 'gaussian', level: forecastNoise, seed: forecastSeed }],
      });
      const forecastField = perturbed.perturbed_field;
      setForecastProvenance({ ...perturbed, seed: forecastSeed });
        const res = await apiService.computeDiagnostics({
          forecast_data: forecastField,
          ground_truth_data: primaryField
        });
        setDiagnosticsResults(res);

        const scaleRes = await apiService.decomposeErrors({
          forecast_data: forecastField,
          ground_truth_data: primaryField,
          boundary_width: 8
        });
        setScaleDecompResults(scaleRes.scale_decomposition || null);
        setBoundaryDecompResults(scaleRes.boundary_decomposition || null);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- TAB 6 ACTIONS: Declarative Experiment Engine ---
  const handleRunExperiment = async () => {
    setLoading(true);
    setError(null);
    setExperimentDetail(null);
    setLineageData(null);
    try {
      const parsedConfig = JSON.parse(experimentJson);
      const res = await apiService.createExperiment(parsedConfig);
      setSubmittedExperimentId(res.id);
      pollExperiment(res.id);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const pollExperiment = async (id: string) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const detail = await apiService.getExperiment(id);
        setExperimentDetail(detail);
        if (detail.status === 'COMPLETED' || detail.status === 'FAILED' || attempts > 10) {
          clearInterval(interval);
          // Load lineage
          const lin = await apiService.getExperimentLineage(id);
          setLineageData(lin);
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 1500);
  };

  // --- TAB 7 ACTIONS: Discover & Proposals ---
  const handleDiscoverHypotheses = async () => {
    setLoading(true);
    setError(null);
    try {
      const discovered = await apiService.discoverHypotheses({
        confidence_threshold: confidenceThreshold
      });
      setHypotheses(discovered);
      const activeProposals = await apiService.getProposals();
      setProposals(activeProposals);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAdoptProposal = (config: any) => {
    setExperimentJson(JSON.stringify(config, null, 2));
    setActiveTab('declarative');
  };


  /**
   * The provenance stamped into every export of the active field.
   *
   * Assembled in one place so a CSV and a NetCDF of the same field cannot disagree, and so
   * `is_simulated` is impossible to omit: a synthetic field is fabricated by definition, and a
   * crop of a dataset inherits whatever that dataset declared.
   */
  const fieldProvenance = (extra: Record<string, any> = {}): Record<string, any> => ({
    origin: primaryMetadata?.type || 'in-memory grid',
    generator_metadata: primaryMetadata || {},
    is_simulated: true,
    simulated_reason:
      'produced by the synthetic field generator, not observed. Any statistic derived from ' +
      'it describes the generator, not the atmosphere.',
    grid_shape: [primaryField.length, primaryField[0]?.length || 0],
    ...extra,
  });

  const datasetProvenance = (): Record<string, any> => {
    const chosen = datasets.find(d => d.id === selectedDatasetId);
    return {
      dataset_id: selectedDatasetId,
      variable: selectedVariable,
      pressure_level: selectedLevel,
      is_simulated: chosen?.is_simulated ?? true,
      fallback_reason: chosen?.fallback_reason ?? null,
      source_kind: chosen?.source_kind ?? 'unknown',
      source_path: chosen?.source_path ?? null,
      spatial_resolution: chosen?.spatial_resolution ?? null,
    };
  };

  // ---------------------------------------------------------------- T3.5.22 handlers

  const loadPlatformStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, b, srcs] = await Promise.all([
        apiService.getHealth(),
        apiService.listBenchmarks(),
        apiService.listDataSources(),
      ]);
      setHealth(h);
      setBenchmarks(b);
      setDataSources(srcs);
    } catch (e: any) {
      setError(`Platform status unavailable: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadEvaluationReports = async () => {
    try {
      setEvaluationReports(await apiService.listEvaluationReports());
    } catch (e: any) {
      setError(`Verified evaluation reports unavailable: ${e.message}`);
    }
  };

  const handleImportEvaluationReceipt = async (file: File) => {
    setReceiptImporting(true);
    setError(null);
    try {
      const imported = await apiService.importEvaluationReceipt(file);
      // Re-read through the normal verified GET boundary as well as displaying the import
      // response; this proves the content-addressed store can reproduce what it admitted.
      const stored = await apiService.getEvaluationReport(imported.report_id);
      setEvaluationReports(current => [stored, ...current.filter(r => r.report_id !== stored.report_id)]);
    } catch (e: any) {
      setError(`Receipt refused: ${e.message}`);
    } finally {
      setReceiptImporting(false);
    }
  };

  const handleRunBenchmarks = async () => {
    setBenchmarkRunning(true);
    setError(null);
    try {
      setBenchmarkRun(await apiService.runBenchmarks(benchmarkSeed));
    } catch (e: any) {
      setError(`Benchmark run failed: ${e.message}`);
    } finally {
      setBenchmarkRunning(false);
    }
  };

  const handleImportedField = (result: types.ImportFieldResponse) => {
    setPrimaryField(result.field_data);
    if (result.coords && Object.keys(result.coords).length) {
      setPrimaryCoords(result.coords as Record<string, number[]>);
    }
    setPrimaryMetadata({ type: `imported: ${result.provenance.filename}`, ...result.provenance });
    setImportedProvenance(result.provenance);
    setPerturbedField(null);
    setPerturbationMetrics(null);
    setError(null);
  };

  useEffect(() => {
    if (activeTab === 'platform' && !health) loadPlatformStatus();
    if (activeTab === 'platform' && registryTransforms.length === 0) {
      Promise.all([apiService.listTransforms(), apiService.listActions()])
        .then(([t, a]) => { setRegistryTransforms(t); setRegistryActions(a); })
        .catch(() => { /* the status load already reports an unreachable backend */ });
    }
    if (activeTab === 'evaluation') loadEvaluationReports();
  }, [activeTab]);

  useEffect(() => {
    if (hasMountedRef.current) workspaceHeadingRef.current?.focus();
    hasMountedRef.current = true;
  }, [activeTab]);

  const activeWorkspace = (WORKFLOW_NAV as readonly {
    items: readonly { id: string; name: string }[];
  }[]).flatMap(group => group.items)
    .find(item => item.id === activeTab)?.name || 'Scientific workbench';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <a href="#workspace-main" className="skip-link">Skip to workspace</a>
      {/* Top Banner / Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Globe className="w-8 h-8 text-teal-400 animate-pulse" aria-hidden="true" />
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">SpectralEarth</h1>
            <p className="text-xs text-slate-400">Scientific Visual Research Workbench</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {backendConnected ? (
            <span role="status" className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-full font-medium flex items-center gap-2">
              <Server className="w-3.5 h-3.5" aria-hidden="true" /> API Connected (SQLite DB Active)
            </span>
          ) : (
            <button type="button" className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 px-3 py-1.5 rounded-full font-medium flex items-center gap-2"
              onClick={checkConnection}>
              <WifiOff className="w-3.5 h-3.5" aria-hidden="true" /> Backend unreachable - no computation available; retry
            </button>
          )}
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Side Navigation bar */}
        <nav aria-label="Scientific workflow"
          className="w-full lg:w-72 border-r border-slate-800 bg-slate-900/10 p-4 space-y-5">
          {WORKFLOW_NAV.map(group => (
            <section key={group.section} aria-labelledby={`nav-${group.section.toLowerCase()}`}>
              <div className="px-3 mb-1">
                <h2 id={`nav-${group.section.toLowerCase()}`}
                  className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  {group.section}
                </h2>
                {'note' in group && group.note && (
                  <p className="text-[10px] text-slate-600 mt-0.5">{group.note}</p>
                )}
              </div>
              <div className="space-y-1">
                {group.items.map(tab => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  const decision = 'operation' in tab && selectedCapability
                    ? selectedCapability.operations[tab.operation] : undefined;
                  const unavailable = decision ? !decision.available : false;
                  return (
                    <button key={tab.id} type="button" disabled={unavailable}
                      onClick={() => setActiveTab(tab.id)}
                      aria-current={isActive ? 'page' : undefined}
                      aria-describedby={unavailable ? `nav-reason-${tab.id}` : undefined}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                        isActive
                          ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20 shadow-sm shadow-teal-500/5'
                          : unavailable ? 'text-slate-600 cursor-not-allowed border border-slate-900'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                      }`}>
                      <Icon className={`w-4 h-4 ${isActive ? 'text-teal-400' : 'text-slate-400'}`}
                        aria-hidden="true" />
                      <span className="min-w-0 text-left">
                        <span className="block">{tab.name}</span>
                        {'context' in tab && (
                          <span className="block text-[9px] uppercase tracking-wide text-slate-600">
                            {tab.context}
                          </span>
                        )}
                        {unavailable && decision && <span id={`nav-reason-${tab.id}`}
                          className="block text-[10px] font-normal leading-tight text-amber-500/80 mt-1">
                          Unavailable — {decision.reason}
                        </span>}
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}

          <div className="pt-6 px-3 border-t border-slate-800 mt-6">
            <span className="text-xs text-slate-500 uppercase font-semibold block mb-2">Gridded field buffer</span>
            <div className="bg-slate-900/50 p-3 border border-slate-800 rounded-lg text-xs space-y-1 text-slate-400">
              <p><strong className="text-slate-300">Dimensions:</strong> {primaryField.length} x {primaryField[0]?.length || 0}</p>
              <p><strong className="text-slate-300">Origin:</strong> {primaryMetadata.type || 'In-Memory Grid'}</p>
              <p><strong className="text-slate-300">Min/Max:</strong> {Math.min(...primaryField.flat()).toFixed(3)} / {Math.max(...primaryField.flat()).toFixed(3)}</p>
            </div>
          </div>
        </nav>

        {/* Core Main content section */}
        <main id="workspace-main" aria-labelledby="workspace-heading"
          aria-busy={loading || benchmarkRunning || receiptImporting}
          className="flex-1 p-6 overflow-y-auto space-y-6">
          <h2 id="workspace-heading" ref={workspaceHeadingRef} tabIndex={-1} className="sr-only">
            {activeWorkspace} workspace
          </h2>
          <p className="sr-only" role="status" aria-live="polite">
            {loading || benchmarkRunning || receiptImporting ? `${activeWorkspace} is working` : `${activeWorkspace} is ready`}
          </p>
          <section aria-label="Current research context"
            className="bg-slate-900/60 border border-slate-800 rounded-lg px-4 py-3 flex flex-wrap gap-x-6 gap-y-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-wide text-slate-500">Record</span>
              {selectedRecord ? <>
                <span className="text-slate-200">{selectedRecord.record.source_name}</span>
                <span className="text-teal-400">{selectedRecord.record.domain}</span>
                <span className="font-mono text-slate-600">
                  {selectedRecord.record.content_sha256.slice(0, 12)}…
                </span>
                <button type="button" onClick={() => { setSelectedRecord(null); setSelectedCapability(null); }}
                  aria-label="Clear selected record" className="text-slate-500 hover:text-slate-200">×</button>
              </> : <span className="text-slate-600">none selected</span>}
            </div>
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-wide text-slate-500">Study</span>
              {selectedStudyId ? <>
                <span className="font-mono text-slate-200">{selectedStudyId}</span>
                <button type="button" onClick={() => setSelectedStudyId('')}
                  aria-label="Clear selected study" className="text-slate-500 hover:text-slate-200">×</button>
              </> : <span className="text-slate-600">none selected</span>}
            </div>
          </section>
          {selectedCapability && <DatasetCapabilityProfile profile={selectedCapability} compact />}
          {error && (
            <div role="alert" className="bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <XCircle className="w-5 h-5 flex-shrink-0" aria-hidden="true" />
                <p className="text-sm font-medium">{error}</p>
              </div>
              <button type="button" onClick={() => setError(null)} className="text-xs underline hover:text-rose-200">Dismiss error</button>
            </div>
          )}

          {/* TAB 1: SYNTHETIC FIELD GENERATION */}
          {activeTab === 'synthetic' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Layers className="text-teal-400 w-5 h-5" /> Synthetic Field Generator
                </h2>
                <p className="text-sm text-slate-400">Generate deterministic physical fields (sinusoids, vortices, fronts) based on analytical equations to benchmark transforms.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Gen Config Controller */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-2">
                    <SlidersHorizontal className="w-4 h-4 text-teal-400" /> Analytical Coordinates
                  </h3>

                  <div>
                    <label htmlFor="synthetic-field-type" className="text-xs text-slate-400 block mb-1">Field Type</label>
                    <select
                      id="synthetic-field-type"
                      value={genType}
                      onChange={(e) => setGenType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-teal-500"
                    >
                      <option value="sinusoid">2D Sinusoid Lattice</option>
                      <option value="vortex">Gaussian Core Vortex Plume</option>
                      <option value="front">Hyperbolic Tangent Front (tanh)</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="synthetic-grid-size" className="text-xs text-slate-400 block mb-1">Grid Size (N x N)</label>
                    <select
                      id="synthetic-grid-size"
                      value={gridSize}
                      onChange={(e) => setGridSize(Number(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      <option value={16}>16 x 16 (Coarse)</option>
                      <option value={32}>32 x 32 (Standard)</option>
                      <option value={64}>64 x 64 (Dense)</option>
                      <option value={128}>128 x 128 (Ultra)</option>
                    </select>
                  </div>

                  {genType === 'sinusoid' && (
                    <div className="space-y-3">
                      <div>
                        <label htmlFor="synthetic-frequency-x" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>X-Frequency (ωx): {freqX}</span>
                        </label>
                        <input
                          id="synthetic-frequency-x"
                          type="range" min="0.5" max="8" step="0.5" value={freqX}
                          onChange={(e) => setFreqX(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="synthetic-frequency-y" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Y-Frequency (ωy): {freqY}</span>
                        </label>
                        <input
                          id="synthetic-frequency-y"
                          type="range" min="0.5" max="8" step="0.5" value={freqY}
                          onChange={(e) => setFreqY(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="synthetic-amplitude" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {sinAmplitude}</span>
                        </label>
                        <input
                          id="synthetic-amplitude"
                          type="range" min="0.5" max="5" step="0.5" value={sinAmplitude}
                          onChange={(e) => setSinAmplitude(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                    </div>
                  )}

                  {genType === 'vortex' && (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label htmlFor="vortex-center-x" className="text-xs text-slate-400 block mb-1">Center X</label>
                          <input
                            id="vortex-center-x"
                            type="number" step="0.1" value={vortexCenterX}
                            onChange={(e) => setVortexCenterX(parseFloat(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                          />
                        </div>
                        <div>
                          <label htmlFor="vortex-center-y" className="text-xs text-slate-400 block mb-1">Center Y</label>
                          <input
                            id="vortex-center-y"
                            type="number" step="0.1" value={vortexCenterY}
                            onChange={(e) => setVortexCenterY(parseFloat(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                          />
                        </div>
                      </div>
                      <div>
                        <label htmlFor="vortex-amplitude" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {vortexAmp}</span>
                        </label>
                        <input
                          id="vortex-amplitude"
                          type="range" min="0.5" max="5" step="0.1" value={vortexAmp}
                          onChange={(e) => setVortexAmp(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="vortex-core-radius" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>{"Core Radius (r_core): "}{vortexRadius}</span>
                        </label>
                        <input
                          id="vortex-core-radius"
                          type="range" min="0.05" max="0.5" step="0.05" value={vortexRadius}
                          onChange={(e) => setVortexRadius(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                    </div>
                  )}

                  {genType === 'front' && (
                    <div className="space-y-3">
                      <div>
                        <label htmlFor="front-angle" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Angle (Degrees): {frontAngle}°</span>
                        </label>
                        <input
                          id="front-angle"
                          type="range" min="0" max="360" step="15" value={frontAngle}
                          onChange={(e) => setFrontAngle(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="front-offset" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Offset: {frontOffset}</span>
                        </label>
                        <input
                          id="front-offset"
                          type="range" min="-0.5" max="0.5" step="0.1" value={frontOffset}
                          onChange={(e) => setFrontOffset(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="front-width" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Width Parameter: {frontWidth}</span>
                        </label>
                        <input
                          id="front-width"
                          type="range" min="0.01" max="0.3" step="0.01" value={frontWidth}
                          onChange={(e) => setFrontWidth(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="front-amplitude" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {frontAmp}</span>
                        </label>
                        <input
                          id="front-amplitude"
                          type="range" min="0.5" max="5" step="0.5" value={frontAmp}
                          onChange={(e) => setFrontAmp(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleGenerateSynthetic}
                    disabled={loading}
                    className="w-full bg-teal-600 hover:bg-teal-500 disabled:bg-slate-800 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition"
                  >
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    Generate Analytical Field
                  </button>
                </div>

                {/* Main Visualizer */}
                <div className="xl:col-span-2 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Heatmap2D data={primaryField} title="Generated Clean Field (F)" colormap="viridis"
                        coords={primaryCoords} divId="fig-clean-field"
                        xLabel="x (normalised)" yLabel="y (normalised)" />
                      <div className="flex flex-col gap-2 px-1">
                        <FieldExportBar field={primaryField} coords={primaryCoords}
                          metadata={fieldProvenance()} variable="field" name="clean_field"
                          label="Export clean field" onError={setError} />
                        <FigureExport targetId="fig-clean-field" name="clean_field"
                          caption="synthetic - dimensionless" onError={setError} />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Heatmap2D data={perturbedField || primaryField} title="Perturbed Spatial Field (F')"
                        colormap="viridis" coords={primaryCoords} divId="fig-perturbed-field"
                        xLabel="x (normalised)" yLabel="y (normalised)" />
                      <div className="flex flex-col gap-2 px-1">
                        <FieldExportBar field={perturbedField} coords={primaryCoords}
                          metadata={fieldProvenance({ perturbations, reproducible: perturbations.every(pp => pp.type !== 'noise' || pp.seed != null) })}
                          variable="field" name="perturbed_field"
                          label="Export perturbed field" onError={setError} />
                        <FigureExport targetId="fig-perturbed-field" name="perturbed_field"
                          caption="synthetic + perturbation" onError={setError} />
                      </div>
                    </div>
                  </div>

                  {/* Perturbation Engine Steps */}
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
                    <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-4 flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-teal-400" /> Sequential Perturbation Scheduler
                    </h3>
                    
                    <div className="flex flex-wrap items-end gap-4 mb-4">
                      <div>
                        <label htmlFor="perturbation-type" className="text-xs text-slate-400 block mb-1">Perturbation Type</label>
                        <select
                          id="perturbation-type"
                          value={newPertType}
                          onChange={(e) => setNewPertType(e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none"
                        >
                          <option value="rotation">Affine Rotation</option>
                          <option value="translation">Affine Translation</option>
                          <option value="noise">Noise Perturbations</option>
                        </select>
                      </div>

                      {newPertType === 'rotation' && (
                        <div>
                          <label htmlFor="perturbation-angle" className="text-xs text-slate-400 block mb-1">Rotation Angle (°)</label>
                          <input
                            id="perturbation-angle"
                            type="number" value={newPertAngle}
                            onChange={(e) => setNewPertAngle(parseFloat(e.target.value))}
                            className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 w-24"
                          />
                        </div>
                      )}

                      {newPertType === 'translation' && (
                        <div className="flex gap-2">
                          <div>
                            <label htmlFor="perturbation-shift-x" className="text-xs text-slate-400 block mb-1">Shift X</label>
                            <input
                              id="perturbation-shift-x"
                              type="number" step="0.05" value={newPertShiftX}
                              onChange={(e) => setNewPertShiftX(parseFloat(e.target.value))}
                              className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 w-20"
                            />
                          </div>
                          <div>
                            <label htmlFor="perturbation-shift-y" className="text-xs text-slate-400 block mb-1">Shift Y</label>
                            <input
                              id="perturbation-shift-y"
                              type="number" step="0.05" value={newPertShiftY}
                              onChange={(e) => setNewPertShiftY(parseFloat(e.target.value))}
                              className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 w-20"
                            />
                          </div>
                        </div>
                      )}

                      {newPertType === 'noise' && (
                        <div className="flex gap-2">
                          <div>
                            <label htmlFor="perturbation-noise-type" className="text-xs text-slate-400 block mb-1">Noise Distribution</label>
                            <select
                              id="perturbation-noise-type"
                              value={newPertNoiseType}
                              onChange={(e) => setNewPertNoiseType(e.target.value)}
                              className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200"
                            >
                              <option value="gaussian">Gaussian</option>
                              <option value="uniform">Uniform</option>
                              <option value="salt_pepper">Salt & Pepper</option>
                            </select>
                          </div>
                          <div>
                            <label htmlFor="perturbation-noise-level" className="text-xs text-slate-400 block mb-1">Level (σ)</label>
                            <input
                              id="perturbation-noise-level"
                              type="number" step="0.05" value={newPertNoiseLevel}
                              onChange={(e) => setNewPertNoiseLevel(parseFloat(e.target.value))}
                              className="w-full accent-teal-500"
                            />
                          </div>
                        </div>
                      )}

                      <button
                        onClick={addPerturbation}
                        className="bg-slate-800 hover:bg-slate-750 text-teal-400 font-semibold text-xs py-2 px-3 rounded-lg flex items-center gap-1 border border-teal-500/10"
                      >
                        <Plus className="w-3.5 h-3.5" /> Add Step
                      </button>
                    </div>

                    {/* Step schedule queue */}
                    {perturbations.length > 0 ? (
                      <div className="space-y-2 mb-4">
                        <span className="text-xs font-semibold text-slate-400 block">Scheduled Steps:</span>
                        <div className="flex flex-wrap gap-2">
                          {perturbations.map((p, idx) => (
                            <span key={idx} className="bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg text-xs flex items-center gap-2">
                              <span className="text-teal-400 font-semibold">{idx+1}. {p.type.toUpperCase()}</span>
                              {p.type === 'rotation' && <span>(θ: {p.angle}°)</span>}
                              {p.type === 'translation' && <span>(X: {p.shift_x}, Y: {p.shift_y})</span>}
                              {p.type === 'noise' && <span>({p.noise_type}, σ: {p.level})</span>}
                              <button type="button" aria-label={`Remove ${p.type} perturbation step ${idx + 1}`}
                                onClick={() => setPerturbations(perturbations.filter((_, i) => i !== idx))} className="text-rose-500 hover:text-rose-400">
                                <Trash2 className="w-3 h-3" aria-hidden="true" />
                              </button>
                            </span>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500 italic mb-4">No perturbations scheduled. Generating will visualize clean field.</p>
                    )}

                    <div className="flex gap-4">
                      <button
                        onClick={handleApplyPerturbations}
                        disabled={loading || perturbations.length === 0}
                        className="flex-1 bg-teal-600/20 text-teal-400 border border-teal-500/30 font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 hover:bg-teal-600/30 transition disabled:opacity-50"
                      >
                        <Play className="w-4 h-4" /> Evaluate Perturbations
                      </button>
                      <button
                        onClick={() => { setPerturbations([]); setPerturbedField(null); setPerturbationMetrics(null); }}
                        className="bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-400 py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                      >
                        <RotateCcw className="w-4 h-4" /> Reset Scheduler
                      </button>
                    </div>

                    {perturbationMetrics && (
                      <div className="mt-5 border-t border-slate-800 pt-4">
                        <span className="text-xs font-semibold text-slate-300 block mb-3">Field Sensitivity Diagnostics</span>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                          {[
                            { name: 'MSE', value: perturbationMetrics.mean_squared_error.toFixed(5) },
                            { name: 'RMSE', value: perturbationMetrics.root_mean_squared_error.toFixed(4) },
                            { name: 'PSNR', value: `${perturbationMetrics.peak_signal_to_noise_ratio.toFixed(2)} dB` },
                            { name: 'SSIM Index', value: perturbationMetrics.structural_similarity_index.toFixed(3) },
                            { name: 'Spectral Shift', value: perturbationMetrics.spectral_energy_shift.toFixed(5) },
                          ].map((m, i) => (
                            <div key={i} className="bg-slate-950 border border-slate-800 p-3 rounded-lg text-center">
                              <span className="text-[10px] text-slate-500 uppercase block">{m.name}</span>
                              <span className="text-sm font-semibold text-teal-400 font-mono mt-1 block">{m.value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* TAB 2: METEOROLOGICAL DATA EXPLORER */}
          {activeTab === 'meteorological' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Wind className="text-teal-400 w-5 h-5" /> Meteorological Data Explorer
                </h2>
                <p className="text-sm text-slate-400">Slice, crop, and visualize simulated or live planetary reanalysis grids (ERA5, GFS, climate models) inside the target physical core.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Spatial Dataset Slicer
                  </h3>

                  <div>
                    <label htmlFor="active-dataset" className="text-xs text-slate-400 block mb-1">Active Dataset</label>
                    <select
                      id="active-dataset"
                      value={selectedDatasetId}
                      onChange={(e) => {
                        const dId = e.target.value;
                        setSelectedDatasetId(dId);
                        const found = datasets.find(d => d.id === dId);
                        if (found) {
                          setSelectedVariable(found.variables[0]);
                          setSelectedLevel(found.pressure_levels?.[0] || 500);
                        }
                      }}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      {datasets.map(d => (
                        <option key={d.id} value={d.id}>
                          {d.name}{d.is_simulated ? '  [SIMULATED]' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <FieldImport onLoaded={handleImportedField} onError={setError} />

                  {importedProvenance && (
                    <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-[10px] font-mono text-slate-400 space-y-1">
                      <div className="text-slate-300 font-sans font-semibold text-[11px]">Active field: imported</div>
                      <div className="flex justify-between"><span>file</span><span className="text-slate-300">{importedProvenance.filename}</span></div>
                      <div className="flex justify-between"><span>variable</span><span className="text-slate-300">{importedProvenance.variable}</span></div>
                      <div className="flex justify-between"><span>sha256</span><span className="text-slate-300">{String(importedProvenance.content_hash).slice(0, 16)}…</span></div>
                      {Object.keys(importedProvenance.selection || {}).length > 0 && (
                        <div className="flex justify-between">
                          <span>slice</span>
                          <span className="text-slate-300">
                            {Object.entries(importedProvenance.selection).map(([k, v]) => `${k}=${v}`).join(' ')}
                          </span>
                        </div>
                      )}
                      {/* null, not false: the platform did not produce this file and asserting
                          it is observational would be inventing a fact. */}
                      <div className={`font-sans leading-relaxed pt-1 border-t border-slate-800 ${
                        importedProvenance.is_simulated === true ? 'text-amber-400'
                          : importedProvenance.is_simulated === false ? 'text-emerald-400'
                          : 'text-slate-400'
                      }`}>
                        {importedProvenance.is_simulated === true
                          ? 'The file declares this data is SIMULATED.'
                          : importedProvenance.is_simulated === false
                          ? 'The file declares this data is observational.'
                          : 'Origin unknown - the file carries no provenance, so the platform makes no claim about whether this is real data.'}
                      </div>
                    </div>
                  )}

                  {/* The single most important thing on this tab. The flag is read from what
                      the source *declared*, not inferred from its name or kind - inferring it
                      from `kind == "simulated"` is the defect that reported a demo source as
                      real observational data. */}
                  {(() => {
                    const chosen = datasets.find(d => d.id === selectedDatasetId);
                    if (!chosen) return null;
                    return chosen.is_simulated ? (
                      <div className="bg-amber-500/10 border border-amber-500/25 rounded-lg p-3 flex gap-2 text-[11px] text-amber-300 leading-relaxed">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <p className="font-semibold">SIMULATED DATA - not an observation.</p>
                          {chosen.fallback_reason && <p className="text-amber-400/80">{chosen.fallback_reason}</p>}
                          <p className="text-amber-400/60 font-mono">source kind: {chosen.source_kind || 'unknown'}</p>
                          <p className="text-amber-400/60">Every export of this crop carries the same warning inside the file.</p>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-3 flex gap-2 text-[11px] text-emerald-300 leading-relaxed">
                        <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-semibold">Real observational data.</p>
                          <p className="text-emerald-400/70 font-mono">
                            {chosen.source_kind || 'unknown'}{chosen.source_path ? ` - ${chosen.source_path}` : ''}
                          </p>
                        </div>
                      </div>
                    );
                  })()}

                  <div>
                    <label htmlFor="dataset-variable" className="text-xs text-slate-400 block mb-1">Variable</label>
                    <select
                      id="dataset-variable"
                      value={selectedVariable}
                      onChange={(e) => setSelectedVariable(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      {datasets.find(d => d.id === selectedDatasetId)?.variables.map(v => (
                        <option key={v} value={v}>{v.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  {datasets.find(d => d.id === selectedDatasetId)?.pressure_levels && (
                    <div>
                      <label htmlFor="dataset-pressure-level" className="text-xs text-slate-400 block mb-1">Pressure Level (hPa)</label>
                      <select
                        id="dataset-pressure-level"
                        value={selectedLevel}
                        onChange={(e) => setSelectedLevel(Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                      >
                        {datasets.find(d => d.id === selectedDatasetId)?.pressure_levels?.map(lvl => (
                          <option key={lvl} value={lvl}>{lvl} hPa</option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="space-y-2">
                    <span className="text-xs text-slate-400 block font-semibold">Geographical Crop Coordinates</span>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label htmlFor="latitude-min" className="text-[10px] text-slate-500 block">Latitude Min</label>
                        <input
                          id="latitude-min"
                          type="number" value={latMin} onChange={(e) => setLatMin(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                      <div>
                        <label htmlFor="latitude-max" className="text-[10px] text-slate-500 block">Latitude Max</label>
                        <input
                          id="latitude-max"
                          type="number" value={latMax} onChange={(e) => setLatMax(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label htmlFor="longitude-min" className="text-[10px] text-slate-500 block">Longitude Min</label>
                        <input
                          id="longitude-min"
                          type="number" value={lonMin} onChange={(e) => setLonMin(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                      <div>
                        <label htmlFor="longitude-max" className="text-[10px] text-slate-500 block">Longitude Max</label>
                        <input
                          id="longitude-max"
                          type="number" value={lonMax} onChange={(e) => setLonMax(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={handleSliceDataset}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Database className="w-4 h-4" /> Slice & Load Field
                  </button>
                </div>

                <div className="xl:col-span-2 space-y-4">
                  {slicedField ? (
                    <div className="space-y-4">
                      <Heatmap2D data={slicedField}
                        title={`${selectedVariable.toUpperCase()} Crop (${selectedDatasetId})`}
                        coords={slicedCoords} colormap="viridis" divId="fig-dataset-crop"
                        xLabel="longitude (degrees east)" yLabel="latitude (degrees north)" />
                      <div className="flex flex-col gap-2 px-1 mt-2">
                        <FieldExportBar field={slicedField} coords={slicedCoords}
                          metadata={datasetProvenance()} variable={selectedVariable}
                          name={`${selectedDatasetId}_${selectedVariable}`}
                          label="Export crop" onError={setError} />
                        <FigureExport targetId="fig-dataset-crop"
                          name={`${selectedDatasetId}_${selectedVariable}`}
                          caption={datasets.find(d => d.id === selectedDatasetId)?.is_simulated
                            ? 'SIMULATED - not an observation'
                            : 'observational'}
                          onError={setError} />
                      </div>
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs space-y-2 text-slate-400">
                        <span className="text-slate-300 font-semibold block mb-1">Metadata Summary</span>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <span className="text-slate-500 block">Resolution</span>
                            <span className="text-slate-200 font-semibold">{datasets.find(d => d.id === selectedDatasetId)?.spatial_resolution}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Variables Available</span>
                            <span className="text-slate-200 font-semibold">{datasets.find(d => d.id === selectedDatasetId)?.variables.join(', ').toUpperCase()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Dimensions Extracted</span>
                            <span className="text-slate-200 font-semibold">{slicedField.length} lat x {slicedField[0].length} lon</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Source Engine</span>
                            <span className="text-slate-200 font-semibold">xarray Dataset</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-500 flex flex-col items-center justify-center h-full min-h-[300px]">
                      <Wind className="w-12 h-12 text-slate-750 mb-3" />
                      <p className="text-sm font-semibold text-slate-400">No active Meteorological slice selected</p>
                      <p className="text-xs text-slate-500 mt-1 max-w-sm">Configure variables and coordinate parameters in the controller, then slice to run local reanalysis grids.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: BOUNDARY-CONDITION LAB */}
          {activeTab === 'boundary' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Sliders className="text-teal-400 w-5 h-5" /> Boundary-Condition Laboratory
                </h2>
                <p className="text-sm text-slate-400">Examine how boundary treatments (periodic, reflection, zero, replicate) and tapering spectral windows (Hann, Tukey) affect spatial grids and limit high-frequency leaks.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Boundary Settings
                  </h3>

                  <div>
                    <label htmlFor="boundary-treatment" className="text-xs text-slate-400 block mb-1">Padding Treatment</label>
                    <select
                      id="boundary-treatment"
                      value={boundaryTreatment}
                      onChange={(e) => setBoundaryTreatment(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      <option value="reflect">Reflective Padding (Mirror)</option>
                      <option value="periodic">Periodic Padding (Circular)</option>
                      <option value="zero">Zero Constant Padding</option>
                      <option value="replicate">Edge Replication</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="boundary-pad-width" className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Pad Width: {padWidth}px</span>
                    </label>
                    <input
                      id="boundary-pad-width"
                      type="range" min="1" max="16" step="1" value={padWidth}
                      onChange={(e) => setPadWidth(parseInt(e.target.value))}
                      className="w-full accent-teal-500"
                    />
                  </div>

                  <div>
                    <label htmlFor="boundary-window" className="text-xs text-slate-400 block mb-1">Spectral Window Tapering</label>
                    <select
                      id="boundary-window"
                      value={windowType}
                      onChange={(e) => setWindowType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      <option value="tukey">Tukey Window (Cosine Taper)</option>
                      <option value="hann">Hann Window</option>
                      <option value="hamming">Hamming Window</option>
                      <option value="none">None (Rectangular Boxcar)</option>
                    </select>
                  </div>

                  {windowType === 'tukey' && (
                    <div>
                      <label htmlFor="boundary-window-alpha" className="text-xs text-slate-400 flex justify-between mb-1">
                        <span>Tukey Alpha (α): {windowAlpha}</span>
                      </label>
                      <input
                        id="boundary-window-alpha"
                        type="range" min="0" max="1" step="0.05" value={windowAlpha}
                        onChange={(e) => setWindowAlpha(parseFloat(e.target.value))}
                        className="w-full accent-teal-500"
                      />
                    </div>
                  )}

                  <button
                    onClick={handleAnalyzeBoundary}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Sliders className="w-4 h-4" /> Apply & Analyze Artefacts
                  </button>
                </div>

                <div className="xl:col-span-2 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Heatmap2D data={primaryField} title="Original Spatial Domain" colormap="viridis" />
                    <Heatmap2D data={paddedField || primaryField} title="Padded Boundary Domain" colormap="viridis" />
                  </div>

                  {distanceProfiles.length > 0 && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      <div className="lg:col-span-2">
                        <LineChart
                          series={[
                            {
                              name: 'Mean Spatial Gradient Magnitude',
                              x: distanceProfiles.map(p => p.distance),
                              y: distanceProfiles.map(p => p.mean_gradient),
                              color: '#2dd4bf'
                            },
                            {
                              name: 'Mean Abs Error Profiles',
                              x: distanceProfiles.map(p => p.distance),
                              y: distanceProfiles.map(p => p.mean_absolute_error),
                              color: '#a855f7'
                            }
                          ]}
                          title="Artefact Gradients & Error Profiles by Boundary Distance"
                          xLabel="Euclidean Distance from Boundary Grid Edge"
                          yLabel="Mean Value"
                        />
                      </div>

                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
                        <div>
                          <h4 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-3">Boundary Metrics</h4>
                          <p className="text-xs text-slate-400 mb-4 leading-relaxed">Quantifies how boundary conditions leakage into high spatial frequencies in the Fourier transform spectra.</p>
                          <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl text-center">
                            <span className="text-xs text-slate-500 block uppercase">Spectral Leakage Ratio</span>
                            <span className="text-2xl font-bold font-mono text-teal-400 mt-2 block">
                              {spectralLeakage ? spectralLeakage.toFixed(4) : '0.0000'}
                            </span>
                            <span className="text-[10px] text-slate-500 mt-1 block">Value &gt; 1 indicating high frequency artifacts</span>
                          </div>
                        </div>
                        <div className="text-[11px] text-slate-500 leading-relaxed mt-4">
                          Note: Applying a cosine-tapering window like Tukey limits edge discontinuity and drops high-frequency leakage towards normal.
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: SPECTRAL TRANSFORM ENGINE */}
          {activeTab === 'spectral' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Activity className="text-teal-400 w-5 h-5" /> Spectral Transform Engine
                </h2>
                <p className="text-sm text-slate-400">Perform rigorous mathematical forward and inverse transforms (FFT, DCT, multi-level Wavelets, Dual-Tree DTCWT, or low/high Hybrid frequencies) and evaluate reconstructed field accuracy.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Transform Configuration
                  </h3>

                  <div>
                    <label htmlFor="transform-operator" className="text-xs text-slate-400 block mb-1">Transform Operator</label>
                    <select
                      id="transform-operator"
                      value={transformType}
                      onChange={(e) => setTransformType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      <option value="fft">2D Fast Fourier Transform (FFT)</option>
                      <option value="dct">2D Discrete Cosine Transform (DCT)</option>
                      <option value="dwt">2D Haar Discrete Wavelet (DWT)</option>
                      <option value="swt">2D Stationary Wavelet (SWT — shift invariant)</option>
                      <option value="dtcwt">2D Dual-Tree Complex Wavelet (DTCWT)</option>
                      <option value="hybrid">FFT Low-Pass + DWT Residual Hybrid</option>
                    </select>
                  </div>

                  {['dwt', 'swt', 'dtcwt'].includes(transformType) && (
                    <div>
                      <label htmlFor="transform-levels" className="text-xs text-slate-400 flex justify-between mb-1">
                        <span>Wavelet Levels: {waveletLevels}</span>
                      </label>
                      <input
                        id="transform-levels"
                        type="range" min="1" max="4" step="1" value={waveletLevels}
                        onChange={(e) => setWaveletLevels(parseInt(e.target.value))}
                        className="w-full accent-teal-500"
                      />
                    </div>
                  )}

                  {transformType === 'swt' && (
                    <div>
                      <label htmlFor="swt-wavelet-family" className="text-xs text-slate-400 block mb-1">SWT Wavelet Family</label>
                      <select
                        id="swt-wavelet-family"
                        value={waveletFamily}
                        onChange={(e) => setWaveletFamily(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                      >
                        <option value="haar">Haar — shortest support</option>
                        <option value="db2">db2 — poster comparison</option>
                        <option value="db3">db3 — longer/smoother support</option>
                      </select>
                    </div>
                  )}

                  {transformType === 'hybrid' && (
                    <div className="space-y-3">
                      <div>
                        <label htmlFor="hybrid-crossover" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Crossover Frequency: {crossoverFreq}</span>
                        </label>
                        <input
                          id="hybrid-crossover"
                          type="range" min="0.05" max="0.5" step="0.05" value={crossoverFreq}
                          onChange={(e) => setCrossoverFreq(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label htmlFor="hybrid-mixing-weight" className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Mixing Weight: {mixingWeight}</span>
                        </label>
                        <input
                          id="hybrid-mixing-weight"
                          type="range" min="0" max="1" step="0.1" value={mixingWeight}
                          onChange={(e) => setMixingWeight(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleApplyTransform}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Activity className="w-4 h-4" /> Apply Forward & Inverse
                  </button>
                </div>

                <div className="xl:col-span-2 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Heatmap2D data={primaryField} title="Original Target Field (F)" colormap="viridis" />
                    <Heatmap2D data={reconstructedField || primaryField} title="Inverse Reconstructed Field (F-hat)" colormap="viridis" />
                  </div>

                  {transformMetrics && (
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                      <h4 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-4">Reconstruction Accuracy Metrics</h4>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl text-center">
                          <span className="text-xs text-slate-500 block uppercase">Mean Squared Error (MSE)</span>
                          <span className="text-lg font-bold font-mono text-teal-400 mt-2 block">{transformMetrics.mean_squared_error.toExponential(6)}</span>
                        </div>
                        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl text-center">
                          <span className="text-xs text-slate-500 block uppercase">Max Absolute Error</span>
                          <span className="text-lg font-bold font-mono text-teal-400 mt-2 block">{transformMetrics.max_absolute_error.toFixed(6)}</span>
                        </div>
                        {/* These were two unconditional green ticks reading "Mathematically
                            rigorous floating point calculations" and "Verified perfect
                            reconstruct limits" - shown whatever the measured error was, which
                            is an unqualified validation claim of exactly the kind the project's
                            own rules forbid in its documents. Replaced by the measurement,
                            judged against a stated tolerance. */}
                        <div className="col-span-2 md:col-span-1 bg-slate-950 border border-slate-800 p-4 rounded-xl flex flex-col justify-center text-xs leading-relaxed">
                          {transformMetrics.max_absolute_error <= 1e-9 ? (
                            <p className="flex items-start gap-1.5 text-emerald-400">
                              <CheckCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                              <span>Round trip within 1e-9 &mdash; perfect reconstruction to double precision.</span>
                            </p>
                          ) : transformMetrics.max_absolute_error <= 1e-5 ? (
                            <p className="flex items-start gap-1.5 text-amber-400">
                              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                              <span>Round-trip error {transformMetrics.max_absolute_error.toExponential(2)} exceeds 1e-9. Expected for a lossy or non-tight-frame configuration; not expected for fft/dct/swt.</span>
                            </p>
                          ) : (
                            <p className="flex items-start gap-1.5 text-rose-400">
                              <XCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                              <span>Round-trip error {transformMetrics.max_absolute_error.toExponential(2)} is too large to treat this reconstruction as faithful.</span>
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  {transformType === 'dtcwt' && (
                    <DTCWTScientificView summary={transformCoefficients as any} />
                  )}
                </div>
              </div>

              <TrainingReadiness catalogue={trainingCatalogue} />
            </div>
          )}

          {/* TAB 5: DIAGNOSTIC & ANALYSIS */}
          {activeTab === 'analysis' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <BarChart2 className="text-teal-400 w-5 h-5" /> Diagnostic & Analysis Engine
                </h2>
                <p className="text-sm text-slate-400">Perform complete spatial error benchmarking of weather forecasts against physical ground truths, decompose error across frequency scales, and analyze cross-spectral coherence.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Spatial Noise Config
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Builds a synthetic &quot;forecast&quot; by adding <strong className="text-slate-300">seeded
                    Gaussian</strong> noise to the active field, through the backend&apos;s perturbation
                    engine. The seed is explicit because a diagnostic whose input cannot be
                    regenerated is not a measurement of anything.
                  </p>

                  <div>
                    <label htmlFor="diagnostic-noise-seed" className="text-xs text-slate-400 block mb-1">Noise seed</label>
                    <div className="flex gap-2">
                      <input
                        id="diagnostic-noise-seed"
                        type="number"
                        value={forecastSeed}
                        onChange={(e) => setForecastSeed(parseInt(e.target.value, 10) || 0)}
                        className="flex-1 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 font-mono"
                      />
                      <button
                        type="button"
                        aria-label="Draw a new diagnostic noise seed"
                        onClick={() => setForecastSeed(Math.floor(Math.random() * 2147483647))}
                        title="Draw a new seed. The value is recorded, so the run stays reproducible."
                        className="bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-400 px-2 rounded"
                      >
                        <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                      </button>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="diagnostic-noise-level" className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Forecast Noise StDev: {forecastNoise}</span>
                    </label>
                    <input
                      id="diagnostic-noise-level"
                      type="range" min="0.05" max="0.5" step="0.05" value={forecastNoise}
                      onChange={(e) => setForecastNoise(parseFloat(e.target.value))}
                      className="w-full accent-teal-500"
                    />
                  </div>

                  <button
                    onClick={handleComputeDiagnostics}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <BarChart2 className="w-4 h-4" /> Run Benchmarking diagnostics
                  </button>
                </div>

                <div className="xl:col-span-2 space-y-6">
                  {diagnosticsResults ? (
                    <div className="space-y-6">
                      {/* Metric grids */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[
                          { name: 'RMSE', val: diagnosticsResults.spatial_metrics.root_mean_squared_error.toFixed(4) },
                          { name: 'Pearson r', val: diagnosticsResults.spatial_metrics.pearson_correlation.toFixed(4) },
                          { name: 'SSIM Index', val: diagnosticsResults.spatial_metrics.structural_similarity_index.toFixed(3) },
                          { name: 'Grad Mag MAE', val: diagnosticsResults.gradient_errors.gradient_magnitude_mae.toFixed(4) },
                        ].map((m, i) => (
                          <div key={i} className="bg-slate-900 border border-slate-800 p-4 rounded-xl text-center">
                            <span className="text-[10px] text-slate-500 block uppercase font-medium">{m.name}</span>
                            <span className="text-lg font-bold text-teal-400 font-mono mt-1.5 block">{m.val}</span>
                          </div>
                        ))}
                      </div>

                      {/* Coherence line chart */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <LineChart
                          series={[
                            {
                              name: 'Forecast PSD',
                              x: diagnosticsResults.spectral_diagnostics.wavenumbers,
                              y: diagnosticsResults.spectral_diagnostics.forecast_psd,
                              color: '#38bdf8'
                            },
                            {
                              name: 'Ground Truth PSD',
                              x: diagnosticsResults.spectral_diagnostics.wavenumbers,
                              y: diagnosticsResults.spectral_diagnostics.ground_truth_psd,
                              color: '#2dd4bf'
                            }
                          ]}
                          title="Power Spectral Density (PSD)"
                          xLabel="Wavenumber (k)"
                          yLabel="Energy Density"
                        />
                        <LineChart
                          series={[
                            {
                              name: 'Spectral Coherence',
                              x: diagnosticsResults.spectral_diagnostics.wavenumbers,
                              y: diagnosticsResults.spectral_diagnostics.spectral_coherence,
                              color: '#a855f7'
                            }
                          ]}
                          title="Cross-Spectral Coherence"
                          xLabel="Wavenumber (k)"
                          yLabel="Coherence Ratio"
                        />
                      </div>

                      {/* Turbulence Spectral Slope Fits */}
                      {/* Units and the spectral convention, which the backend has returned
                          since T3.5.13 and nothing displayed. R15 requires the convention to be
                          stated wherever a slope is: the same field has different exponents
                          under E(k) and S(k), and an unlabelled beta is not interpretable. */}
                      <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 mb-4 text-[11px] font-mono text-slate-400 space-y-1">
                        <div className="flex justify-between">
                          <span className="text-slate-500">wavenumber units</span>
                          <span className="text-slate-300">{diagnosticsResults.spectral_diagnostics.k_units || 'unlabelled'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">power units</span>
                          <span className="text-slate-300">{diagnosticsResults.spectral_diagnostics.power_units || 'unlabelled'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">convention</span>
                          <span className="text-teal-400">{diagnosticsResults.spectral_diagnostics.convention || 'unstated'}</span>
                        </div>
                        {diagnosticsResults.spectral_diagnostics.convention_note && (
                          <p className="text-slate-500 pt-1 border-t border-slate-800 font-sans leading-relaxed">
                            {diagnosticsResults.spectral_diagnostics.convention_note}
                          </p>
                        )}
                        {diagnosticsResults.grid && (
                          <p className="text-slate-500 font-sans">grid: {diagnosticsResults.grid.description}</p>
                        )}
                        {(diagnosticsResults.spectral_diagnostics.warnings || []).map((w: string, i: number) => (
                          <p key={i} className="text-amber-400 font-sans leading-relaxed">{w}</p>
                        ))}
                      </div>

                      {forecastProvenance && (
                        <div className={`rounded-lg p-3 mb-4 text-[11px] border ${
                          forecastProvenance.reproducible
                            ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-300'
                            : 'bg-amber-500/10 border-amber-500/25 text-amber-300'
                        }`}>
                          {forecastProvenance.reproducible ? (
                            <span>Synthetic forecast drawn with seed <strong className="font-mono">{forecastProvenance.seed}</strong> - re-running this reproduces it exactly.</span>
                          ) : (
                            <span>This synthetic forecast ran unseeded and cannot be reproduced.</span>
                          )}
                        </div>
                      )}

                      {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis && diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis && (
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-slate-950/50 border border-slate-850 p-4 rounded-lg">
                            <span className="text-xs text-slate-500 uppercase font-bold block mb-1">Forecast Spectral Slope</span>
                            <span className="text-xl font-bold text-sky-400 font-mono">
                              β = {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.slope_beta.toFixed(3)}
                              {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.slope_standard_error != null && (
                                <span className="text-sm text-slate-400"> ± {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.slope_standard_error!.toFixed(3)}</span>
                              )}
                            </span>
                            <span className="text-xs text-slate-400 block mt-1">
                              Power-Law Fit R²: {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.r_squared.toFixed(3)}
                            </span>
                            <span className="text-xs text-teal-400 font-medium block mt-1.5 border-t border-slate-800/40 pt-1.5">
                              {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.regime_interpretation}
                            </span>
                          </div>
                          <div className="bg-slate-950/50 border border-slate-850 p-4 rounded-lg">
                            <span className="text-xs text-slate-500 uppercase font-bold block mb-1">Ground Truth Spectral Slope</span>
                            <span className="text-xl font-bold text-teal-400 font-mono">
                              β = {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.slope_beta.toFixed(3)}
                              {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.slope_standard_error != null && (
                                <span className="text-sm text-slate-400"> ± {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.slope_standard_error!.toFixed(3)}</span>
                              )}
                            </span>
                            <span className="text-xs text-slate-400 block mt-1">
                              Power-Law Fit R²: {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.r_squared.toFixed(3)}
                            </span>
                            <span className="text-xs text-teal-400 font-medium block mt-1.5 border-t border-slate-800/40 pt-1.5">
                              {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.regime_interpretation}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Scale and boundary decomposition charts */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {scaleDecompResults && (
                          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                            <span className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-4 block">
                              Error Scale Decomposition (Fourier space)
                            </span>
                            <div className="space-y-4">
                              {[
                                { name: 'Low Scale (Synoptic Waveforms)', val: scaleDecompResults.low_scale_rmse, color: 'bg-emerald-500' },
                                { name: 'Mid Scale (Mesoscale Patterns)', val: scaleDecompResults.mid_scale_rmse, color: 'bg-blue-500' },
                                { name: 'High Scale (Turbulence / Artifacts)', val: scaleDecompResults.high_scale_rmse, color: 'bg-indigo-500' },
                              ].map((scale, i) => {
                                const max = Math.max(scaleDecompResults.low_scale_rmse, scaleDecompResults.mid_scale_rmse, scaleDecompResults.high_scale_rmse);
                                const pct = (scale.val / (max || 1)) * 100;
                                return (
                                  <div key={i} className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                      <span className="text-slate-400">{scale.name}</span>
                                      <span className="font-semibold text-teal-400 font-mono">RMSE: {scale.val.toFixed(4)}</span>
                                    </div>
                                    <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                                      <div className={`${scale.color} h-full rounded-full transition-all`} style={{ width: `${pct}%` }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {boundaryDecompResults && (
                          <div className="bg-slate-900 border border-slate-800 rounded-xl">
                            <LineChart
                              series={[
                                {
                                  name: 'RMSE Profile',
                                  x: boundaryDecompResults.map(b => b.distance),
                                  y: boundaryDecompResults.map(b => b.rmse),
                                  color: '#f43f5e'
                                },
                                {
                                  name: 'MAE Profile',
                                  x: boundaryDecompResults.map(b => b.distance),
                                  y: boundaryDecompResults.map(b => b.mean_absolute_error),
                                  color: '#f59e0b'
                                }
                              ]}
                              title="Forecast Error Decomposed by Boundary Distance"
                              xLabel="Euclidean Distance from Boundary"
                              yLabel="Error Metric Value"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-500 flex flex-col items-center justify-center h-full min-h-[300px]">
                      <BarChart2 className="w-12 h-12 text-slate-750 mb-3" />
                      <p className="text-sm font-semibold text-slate-400">No active Diagnostics analysis evaluated</p>
                      <p className="text-xs text-slate-500 mt-1 max-w-sm">Press the run diagnostics button above to compute complete spatial and frequency benchmarks.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 6: DECLARATIVE EXPERIMENT ENGINE */}
          {activeTab === 'declarative' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <FileCode className="text-teal-400 w-5 h-5" /> Declarative Experiment Engine
                </h2>
                <p className="text-sm text-slate-400">Launch parameter matrix sweep tests and map provenance nodes automatically inside the relational SQLAlchemy schema.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* JSON Configuration Controller */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-2 flex items-center justify-between">
                    <span>Declarative JSON Editor</span>
                    <Code className="w-4 h-4 text-teal-400" />
                  </h3>

                  <label htmlFor="experiment-definition" className="sr-only">Experiment definition JSON</label>
                  <textarea
                    id="experiment-definition"
                    value={experimentJson}
                    onChange={(e) => setExperimentJson(e.target.value)}
                    rows={18}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-teal-400 focus:outline-none focus:border-teal-500"
                  />

                  <button
                    onClick={handleRunExperiment}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Play className="w-4 h-4" /> Deploy & Run Experiment Sweep
                  </button>
                </div>

                <div className="xl:col-span-2 space-y-6">
                  {experimentDetail ? (
                    <div className="space-y-6 animate-fadeIn">
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                        <div className="flex justify-between items-start border-b border-slate-800 pb-3 mb-4">
                          <div>
                            <span className="text-xs text-slate-500 uppercase block font-semibold">Active Experiment Sweep</span>
                            <h4 className="text-base font-bold text-slate-100">{experimentDetail.name}</h4>
                            <p className="text-xs text-slate-400 mt-0.5">{experimentDetail.description}</p>
                          </div>
                          <span className={`text-xs px-2.5 py-1 rounded font-bold border ${
                            experimentDetail.status === 'COMPLETED'
                              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                              : 'bg-teal-500/10 text-teal-400 border-teal-500/20 animate-pulse'
                          }`}>
                            {experimentDetail.status}
                          </span>
                        </div>

                        {/* Runs List */}
                        <div className="space-y-3">
                          <span className="text-xs font-semibold text-slate-300 block">Expanded Cartesian Runs ({experimentDetail.runs.length})</span>
                          <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto pr-2">
                            {experimentDetail.runs.map((r, idx) => (
                              <div key={idx} className="bg-slate-950 border border-slate-805 p-3 rounded-lg flex items-center justify-between">
                                <div className="text-xs space-y-1">
                                  <div className="flex items-center gap-2">
                                    <span className="text-teal-400 font-semibold font-mono">Run #{idx+1} (ID: {r.id.substring(0,6)})</span>
                                    <span className="text-slate-500">|</span>
                                    <span className="text-slate-300">Params: {JSON.stringify(r.parameters)}</span>
                                  </div>
                                  {r.results && (
                                    <div className="text-slate-500 flex items-center gap-1.5 font-mono">
                                      Result MSE: <span className="text-slate-300 font-semibold">
                                        {r.results.trans_metrics?.mean_squared_error?.toExponential(4) || 'N/A'}
                                      </span>
                                    </div>
                                  )}
                                </div>
                                <span className="text-[10px] bg-slate-900 px-2 py-0.5 border border-slate-800 text-slate-400 rounded">
                                  {r.status}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Lineage Node Provenance Tree */}
                      {lineageData && (
                        <div className="space-y-2">
                          <LineageGraph nodes={lineageData.nodes} edges={lineageData.edges} />
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-500 flex flex-col items-center justify-center h-full min-h-[350px]">
                      <FileCode className="w-12 h-12 text-slate-750 mb-3" />
                      <p className="text-sm font-semibold text-slate-400">No active Experiment sweep running</p>
                      <p className="text-xs text-slate-500 mt-1 max-w-sm">Edit the declarative JSON pipeline sweep schema on the left, then deploy it to the SQLite platform engine.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 7: AUTOMATED HYPOTHESIS & PROPOSALS */}
          {activeTab === 'hypothesis' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Lightbulb className="text-teal-400 w-5 h-5" /> Automated Hypothesis Engine
                </h2>
                <p className="text-sm text-slate-400">Mines metrics in the SQLite runs tables using numerical correlation math (Pearson's r) and categorical optimization to discover pattern proposals.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Hypothesis Discovery Config
                  </h3>

                  <div>
                    <label htmlFor="hypothesis-threshold" className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Pearson Confidence Cutoff: {confidenceThreshold}</span>
                    </label>
                    <input
                      id="hypothesis-threshold"
                      type="range" min="0.1" max="0.9" step="0.05" value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="w-full accent-teal-500"
                    />
                  </div>

                  <button
                    onClick={handleDiscoverHypotheses}
                    className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2"
                  >
                    <Lightbulb className="w-4 h-4" /> Run Automated Mining
                  </button>
                </div>

                <div className="xl:col-span-3 space-y-6">
                  {hypotheses.length > 0 ? (
                    <div className="space-y-6">
                      <div className="flex items-center justify-between mb-3 gap-4 flex-wrap">
                        <span className="text-xs font-semibold text-slate-300">Mined Scientific Hypotheses &amp; Adaptive Proposals ({hypotheses.length})</span>
                        <TableExportBar
                          rows={hypotheses.map(h => ({
                            description: h.description,
                            pattern_type: h.pattern_type,
                            effect_size: h.confidence,
                            p_value: h.p_value ?? null,
                            q_value: h.q_value ?? null,
                            n_tests: h.n_tests ?? null,
                            correction: h.statistics?.correction?.method ?? null,
                            dependence_assumption: h.statistics?.correction?.assumption ?? null,
                            metrics: h.metrics_analyzed.join('; '),
                            parameters: h.parameters_analyzed.join('; '),
                            caveat: h.statistics?.caveat ?? null,
                          }))}
                          metadata={{
                            what: 'hypotheses mined from stored experiment runs',
                            note: 'effect_size is |r| and is NOT evidence on its own; judge by q_value',
                            corrected: hypotheses.some(h => h.q_value != null),
                          }}
                          name="hypotheses"
                          label="Export hypotheses"
                          onError={setError}
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {hypotheses.map((h, idx) => (
                          <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
                            <div>
                              <div className="flex items-center justify-between mb-3">
                                <span className={`text-[10px] font-mono px-2 py-0.5 border rounded font-semibold uppercase ${
                                  h.pattern_type === 'correlation'
                                    ? 'bg-sky-500/10 text-sky-400 border-sky-500/20'
                                    : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                }`}>
                                  {h.pattern_type}
                                </span>
                                <span className="text-xs font-mono text-slate-500">Effect size: {(h.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <p className="text-xs text-slate-200 font-medium leading-relaxed mb-4">{h.description}</p>
                              <div className="text-[10px] font-mono bg-slate-950 p-2 border border-slate-850 rounded text-slate-400 space-y-1 mb-4">
                                <div><strong className="text-slate-300">Metric:</strong> {h.metrics_analyzed.join(', ')}</div>
                                <div><strong className="text-slate-300">Parameter:</strong> {h.parameters_analyzed.join(', ')}</div>
                              </div>

                              {/* Defect D8 made visible. The label above says "effect size", not
                                  "confidence", because that is what it is: |r| alone is what let a
                                  9-run sweep read as nine discoveries. A finding is shown with its
                                  q-value and the correction's dependence assumption, or it is shown
                                  as uncorrected - never silently as though it had been tested. */}
                              {h.q_value != null ? (
                                <div className={`text-[10px] font-mono p-2 border rounded space-y-1 mb-4 ${
                                  h.q_value <= 0.05
                                    ? 'bg-emerald-500/5 border-emerald-500/25 text-emerald-300'
                                    : 'bg-slate-950 border-slate-800 text-slate-400'
                                }`}>
                                  <div className="flex justify-between">
                                    <span>q (corrected)</span>
                                    <strong>{h.q_value < 1e-4 ? h.q_value.toExponential(2) : h.q_value.toFixed(4)}</strong>
                                  </div>
                                  {h.p_value != null && (
                                    <div className="flex justify-between text-slate-500">
                                      <span>p (raw)</span>
                                      <span>{h.p_value < 1e-4 ? h.p_value.toExponential(2) : h.p_value.toFixed(4)}</span>
                                    </div>
                                  )}
                                  {h.n_tests != null && (
                                    <div className="flex justify-between text-slate-500">
                                      <span>family size</span><span>{h.n_tests} tests</span>
                                    </div>
                                  )}
                                  {/* `statistics.correction` is an OBJECT - {method, assumption,
                                      n_tests, min_adjusted} - not a string. Rendering it directly
                                      throws "Objects are not valid as a React child", and neither
                                      `tsc` nor the build catches it because the field is typed
                                      Record<string, any>. The keys are asserted by
                                      test_frontend_contract.py against the real payload. */}
                                  {h.statistics?.correction?.assumption && (
                                    <div className="text-slate-500 pt-1 border-t border-slate-800 leading-relaxed">
                                      {h.statistics.correction.method}: {h.statistics.correction.assumption}
                                    </div>
                                  )}
                                  {Array.isArray(h.statistics?.assumptions) && h.statistics.assumptions.length > 0 && (
                                    <div className="text-slate-500 leading-relaxed">
                                      test assumes: {h.statistics.assumptions.join('; ')}
                                    </div>
                                  )}
                                  {h.statistics?.caveat && (
                                    <div className="text-amber-400/80 leading-relaxed">{h.statistics.caveat}</div>
                                  )}
                                </div>
                              ) : (
                                <div className="text-[10px] p-2 border rounded mb-4 bg-amber-500/5 border-amber-500/25 text-amber-400 leading-relaxed flex gap-2">
                                  <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
                                  <span>
                                    No multiplicity correction was reported for this pattern. An effect size
                                    on its own is not evidence of anything: scanning enough parameter/metric
                                    pairs produces strong correlations from noise by construction.
                                  </span>
                                </div>
                              )}
                            </div>

                            {h.proposed_experiment_config && (
                              <button
                                onClick={() => handleAdoptProposal(h.proposed_experiment_config)}
                                className="w-full bg-teal-600/10 hover:bg-teal-600/20 border border-teal-500/25 text-teal-400 text-xs font-semibold py-1.5 rounded flex items-center justify-center gap-1.5 transition"
                              >
                                <TrendingUp className="w-3.5 h-3.5" /> Adopt Adaptive Follow-up config
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-500 flex flex-col items-center justify-center h-full min-h-[300px]">
                      <Lightbulb className="w-12 h-12 text-slate-750 mb-3" />
                      <p className="text-sm font-semibold text-slate-400">No hypotheses discovered yet</p>
                      <p className="text-xs text-slate-500 mt-1 max-w-sm">Deploy some experiment sweeps in Tab 6 first, then launch automated mining to run Pearson's r pattern discovery.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 8: PLATFORM STATUS & EVIDENCE ------------------------------------------ */}
          {activeTab === 'platform' && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <ShieldCheck className="text-teal-400 w-5 h-5" /> Platform Status &amp; Evidence
                </h2>
                <p className="text-sm text-slate-400">
                  What this deployment actually is, and what it has been proved to get right. A result
                  is only interpretable alongside the device it ran on, the schema that stored it and
                  the benchmarks the platform passes.
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={loadPlatformStatus}
                  className="bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold py-2 px-4 rounded-lg flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" /> Refresh
                </button>
              </div>

              <ExperimentReceiptPanel trustOnly />
              <ExperimentQualificationPanel />

              {health && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                    <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                      Execution Environment
                    </h3>
                    <dl className="text-xs space-y-1.5 text-slate-400 font-mono">
                      <div className="flex justify-between"><dt>device</dt><dd className="text-teal-400">{health.torch_device}</dd></div>
                      <div className="flex justify-between"><dt>cpu cores</dt><dd className="text-slate-300">{health.execution?.cpu_count ?? '-'}</dd></div>
                      <div className="flex justify-between"><dt>torch threads</dt><dd className="text-slate-300">{health.execution?.torch_num_threads ?? '-'}</dd></div>
                      <div className="flex justify-between"><dt>backends</dt><dd className="text-slate-300">{(health.execution?.backends || []).join(', ')}</dd></div>
                      <div className="flex justify-between"><dt>default</dt><dd className="text-slate-300">{health.execution?.default_backend}</dd></div>
                    </dl>
                    {health.execution?.default_backend_rationale && (
                      <p className="text-[11px] text-slate-500 leading-relaxed border-t border-slate-800 pt-2">
                        {health.execution.default_backend_rationale}
                      </p>
                    )}
                  </div>

                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                    <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                      Storage &amp; Schema
                    </h3>
                    <dl className="text-xs space-y-1.5 text-slate-400 font-mono">
                      <div className="flex justify-between"><dt>backend</dt><dd className="text-slate-300">{health.database_url_scheme}</dd></div>
                      <div className="flex justify-between"><dt>journal</dt><dd className="text-slate-300">{health.database_settings?.journal_mode ?? '-'}</dd></div>
                      <div className="flex justify-between"><dt>busy timeout</dt><dd className="text-slate-300">{health.database_settings?.busy_timeout_ms ?? '-'} ms</dd></div>
                      <div className="flex justify-between"><dt>revision</dt><dd className="text-slate-300">{health.schema_state?.revision ?? 'none'}</dd></div>
                      <div className="flex justify-between"><dt>head</dt><dd className="text-slate-300">{health.schema_state?.head ?? '-'}</dd></div>
                    </dl>
                    {health.schema_state?.up_to_date === false && (
                      <div className="text-[11px] bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded p-2 flex gap-2">
                        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                        <span>
                          The database schema is behind the code: {(health.schema_state.pending || []).join(', ')} outstanding.
                          Queries touching new columns will fail. Run <code>alembic upgrade head</code>.
                        </span>
                      </div>
                    )}
                    {health.schema_state?.error && (
                      <div className="text-[11px] bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded p-2">
                        {health.schema_state.error}
                      </div>
                    )}
                  </div>

                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                    <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                      Data Sources (priority order)
                    </h3>
                    <div className="space-y-2">
                      {dataSources.map(src => (
                        <div key={src.name} className="text-[11px] border border-slate-800 rounded p-2 bg-slate-950">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-mono text-slate-200">{src.name}</span>
                            {src.capabilities?.observational === true ? (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">observational</span>
                            ) : (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">SIMULATED</span>
                            )}
                          </div>
                          <p className="text-slate-500 leading-relaxed">{src.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {(registryTransforms.length > 0 || registryActions.length > 0) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {([['Transforms', registryTransforms], ['Pipeline actions', registryActions]] as const).map(([title, entries]) => (
                    <div key={title} className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-2">
                      <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center gap-2">
                        <Boxes className="w-4 h-4 text-slate-400" /> {title} ({entries.length})
                      </h3>
                      {/* Generated from the registries, never hand-listed: a hand-written list
                          goes stale precisely when someone adds an entry - the moment it matters.
                          Capabilities are shown because they are what makes a transform
                          selectable by property rather than by name. */}
                      {entries.map(entry => (
                        <div key={entry.name} className="text-[11px] border border-slate-800 rounded p-2 bg-slate-950 space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono text-slate-200">{entry.name}</span>
                            {entry.node_type && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">{entry.node_type}</span>
                            )}
                          </div>
                          <p className="text-slate-500 leading-relaxed">{entry.description}</p>
                          {entry.capabilities && Object.keys(entry.capabilities).length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(entry.capabilities)
                                .filter(([, v]) => v === true)
                                .map(([k]) => (
                                  <span key={k} className="text-[9px] px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20 font-mono">
                                    {k}
                                  </span>
                                ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              {benchmarks.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center justify-between">
                    <span>Ground-Truth Benchmark Suite ({benchmarks.length} datasets)</span>
                    <TableExportBar
                      rows={benchmarks.map(b => ({
                        name: b.name, kind: b.kind, is_null: b.is_null,
                        gates: b.gates.join('; '), description: b.description,
                        known_answer: b.known_answer,
                      }))}
                      metadata={{ what: 'declared known answers of the Ground-Truth Benchmark Suite' }}
                      name="benchmarks" label="Export" onError={setError}
                    />
                    <span className="text-[11px] font-normal text-slate-500">
                      {benchmarks.filter(b => b.is_null).length} of them are NULL benchmarks - the correct answer is &quot;nothing&quot;
                    </span>
                  </h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    Each dataset has a declared known answer derived from its construction, not from what
                    the platform happens to produce. The null benchmarks are the false-positive floor: a
                    discovery engine that reports a finding on spatially uncorrelated noise is broken, and
                    these are what catch that.
                  </p>
                  {/* The gates were listable and not runnable: a researcher could see what the
                      platform claims to get right but could not make it prove it. NOT_YET_RUNNABLE
                      is counted separately and never folded into PASS - a summary that did so
                      would let "all green" mean "we never looked". */}
                  <div className="flex items-end gap-3 flex-wrap border-b border-slate-800 pb-3">
                    <div>
                      <label htmlFor="benchmark-root-seed" className="text-[10px] uppercase tracking-wider text-slate-500 block mb-1">Root seed</label>
                      <input
                        id="benchmark-root-seed"
                        type="number" value={benchmarkSeed}
                        onChange={(e) => setBenchmarkSeed(parseInt(e.target.value, 10) || 0)}
                        className="bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 font-mono w-36"
                      />
                    </div>
                    <button
                      onClick={handleRunBenchmarks}
                      disabled={benchmarkRunning}
                      className="bg-teal-600 hover:bg-teal-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm font-semibold py-2 px-4 rounded-lg flex items-center gap-2"
                    >
                      {benchmarkRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                      Run the suite
                    </button>
                    {benchmarkRun && (
                      <div className="flex items-center gap-3 text-xs font-mono">
                        <span className="text-emerald-400">{benchmarkRun.passed} PASS</span>
                        <span className={benchmarkRun.failed > 0 ? 'text-rose-400 font-bold' : 'text-slate-500'}>
                          {benchmarkRun.failed} FAIL
                        </span>
                        <span className="text-sky-400">{benchmarkRun.not_yet_runnable} NOT YET RUNNABLE</span>
                        <span className="text-slate-600">seed {benchmarkRun.root_seed}</span>
                      </div>
                    )}
                  </div>

                  {benchmarkRun && benchmarkRun.null_failures.length > 0 && (
                    <div className="bg-rose-500/10 border border-rose-500/25 rounded-lg p-3 text-[11px] text-rose-300 space-y-1">
                      <p className="font-semibold flex items-center gap-2">
                        <XCircle className="w-3.5 h-3.5" /> A NULL benchmark reported a discovery.
                      </p>
                      <p className="text-rose-400/80 leading-relaxed">
                        These datasets contain no structure by construction. A finding on one of
                        them is a false positive in the platform itself, not a result.
                      </p>
                      {benchmarkRun.null_failures.map((f, i) => (
                        <p key={i} className="font-mono text-rose-300/90">{f}</p>
                      ))}
                    </div>
                  )}

                  {benchmarkRun && (
                    <div className="flex justify-end">
                      <TableExportBar
                        rows={benchmarkRun.benchmarks.flatMap(b => b.checks.map(c => ({
                          benchmark: b.name, is_null: b.is_null, stage: c.stage,
                          outcome: c.outcome, detail: c.detail,
                          measured: c.measured ?? null,
                        })))}
                        metadata={{ what: 'Ground-Truth Benchmark Suite run',
                                    root_seed: benchmarkRun.root_seed,
                                    passed: benchmarkRun.passed, failed: benchmarkRun.failed,
                                    not_yet_runnable: benchmarkRun.not_yet_runnable }}
                        name="benchmark_run" label="Export results" onError={setError}
                      />
                    </div>
                  )}

                  <div className="overflow-x-auto">
                    <table className="w-full text-[11px] font-mono">
                      <thead className="text-slate-500 border-b border-slate-800">
                        <tr>
                          <th className="text-left py-1.5 pr-3">dataset</th>
                          <th className="text-left py-1.5 pr-3">kind</th>
                          <th className="text-left py-1.5 pr-3">gates</th>
                          <th className="text-left py-1.5">known answer</th>
                          <th className="text-left py-1.5 pl-3">outcome</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchmarks.map(b => (
                          <tr key={b.name} className="border-b border-slate-850 align-top">
                            <td className="py-1.5 pr-3 text-slate-200">
                              {b.name}
                              {b.is_null && (
                                <span className="ml-2 text-[9px] px-1 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">NULL</span>
                              )}
                            </td>
                            <td className="py-1.5 pr-3 text-slate-500">{b.kind}</td>
                            <td className="py-1.5 pr-3 text-slate-400">{b.gates.join(', ')}</td>
                            <td className="py-1.5 text-slate-500 max-w-md truncate" title={JSON.stringify(b.known_answer)}>
                              {JSON.stringify(b.known_answer)}
                            </td>
                            <td className="py-1.5 pl-3">
                              {(() => {
                                const run = benchmarkRun?.benchmarks.find(x => x.name === b.name);
                                if (!run) return <span className="text-slate-700">-</span>;
                                return (
                                  <div className="space-y-0.5">
                                    {run.checks.map((c, i) => (
                                      <div key={i} className="flex items-center gap-1.5" title={c.detail}>
                                        <span className={
                                          c.outcome === 'PASS' ? 'text-emerald-400'
                                            : c.outcome === 'FAIL' ? 'text-rose-400 font-bold'
                                            : 'text-sky-400'
                                        }>
                                          {c.outcome === 'PASS' ? '✓' : c.outcome === 'FAIL' ? '✗' : '·'}
                                        </span>
                                        <span className="text-slate-500">{c.stage}</span>
                                      </div>
                                    ))}
                                  </div>
                                );
                              })()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 9: DOMAIN-FIRST ACQUISITION (TG10.2) ---------------------------------- */}
          {activeTab === 'acquire' && (
            <AcquisitionView onError={setError}
              selectedRecord={selectedRecord} onSelectRecord={setSelectedRecord}
              onCapability={setSelectedCapability} />
          )}

          {/* TG11.1: the selected full channel record reaches domain_analysis without a write. */}
          {activeTab === 'domainWorkbench' && (
            <DomainAnalysisView selectedRecord={selectedRecord}
              onError={(message) => setError(message)} onAcquire={() => setActiveTab('acquire')} />
          )}

          {/* TG11.2: the generate/confirm split. The panel makes the ordering legible; the
              server is what enforces it — a confirmation against an unsealed or already-spent
              partition is refused whatever this file renders (R18). */}
          {activeTab === 'preregistration' && (
            <PreregistrationView selectedRecord={selectedRecord}
              onError={(message) => setError(message)} onAcquire={() => setActiveTab('acquire')} />
          )}

          {/* TG11.3: the evidence write path. The only writing panel in the application, and
              the only one whose response carries a rung — recomputed by the ladder from the
              chain the server just wrote, never sent by this file (R22). */}
          {activeTab === 'evidence' && (
            <EvidenceView selectedRecord={selectedRecord} studyId={selectedStudyId}
              onStudyId={setSelectedStudyId} onError={(message) => setError(message)} />
          )}

          {/* TG11.4: structure mining. Sits in Analyse rather than Evidence because what it
              produces is a confirmation receipt, and recording one is the write path's job.
              No control here can supply a feature or a tolerance: both are measured by the
              server from an admitted field, so a shape cannot be drawn into a result. */}
          {activeTab === 'mining' && (
            <StructureMiningView studyId={selectedStudyId}
              onError={(message) => setError(message)} />
          )}

          {/* TG11.4b: the cross-domain record. Beside structure mining because it shares the
              same seal store and the same held-out ledger, and in Analyse for the same reason:
              what it produces is a confirmation receipt, and recording one is the write path's
              job. It is the only panel whose lag family is entered in seconds - two native
              clocks have two frame sizes, and a lag in frames of either is unreadable by the
              other domain. Nothing here can resample: the two records are aligned on the
              timestamps they actually share, or the request is refused. */}
          {activeTab === 'crossDomainRecord' && (
            <CrossDomainRecordView studyId={selectedStudyId}
              onError={(message) => setError(message)} />
          )}

          {/* TAB 10: VERIFIED FORECAST EVALUATION ------------------------------------ */}
          {activeTab === 'evaluation' && (
            <EvaluationEvidence reports={evaluationReports} importing={receiptImporting}
              onImport={handleImportEvaluationReceipt} />
          )}

          {activeTab === 'experimentComposer' && (
            <ExperimentComposer onSelectStudy={setSelectedStudyId} onEvidenceHandoff={(studyId) => {
              setSelectedStudyId(studyId); setActiveTab('evidence');
            }} />
          )}

          {/* TAB 11: FINDINGS (TG9.2) -------------------------------------------------
              The cross-domain claim surface. Everything scientific on this tab is a string
              the backend produced; this file passes an error handler and nothing else. */}
          {activeTab === 'findings' && (
            <FindingsView onError={(message) => setError(message)}
              selectedStudyId={selectedStudyId} onSelectStudy={setSelectedStudyId} />
          )}

          {/* TG11.5: recorded-not-reproducible argument. This is a separate workspace from
              Findings so commentary cannot become claim text by layout. It reads immutable
              records only and offers no action that can run a panel or move a rung. */}
          {activeTab === 'review' && (
            <ReviewView selectedStudyId={selectedStudyId} onStudyId={setSelectedStudyId}
              onError={(message) => setError(message)} />
          )}

        </main>
      </div>
    </div>
  );
}
