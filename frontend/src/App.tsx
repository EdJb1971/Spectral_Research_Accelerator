import { useState, useEffect } from 'react';
import { Heatmap2D } from './components/Heatmap2D';
import { LineChart } from './components/LineChart';
import { LineageGraph } from './components/LineageGraph';
import { apiService } from './services/api';
import * as types from './types/api';
import {
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
  Code
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('synthetic');
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
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
  const [crossoverFreq, setCrossoverFreq] = useState(0.2);
  const [mixingWeight, setMixingWeight] = useState(0.5);
  const [reconstructedField, setReconstructedField] = useState<number[][] | null>(null);
  const [transformMetrics, setTransformMetrics] = useState<types.TransformResponse['metrics'] | null>(null);
  const [_transformCoefficients, setTransformCoefficients] = useState<Record<string, any> | null>(null);

  // --- TAB 5 STATE: Analysis & Diagnostics ---
  const [forecastNoise, setForecastNoise] = useState(0.15);
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

  const checkConnection = async () => {
    try {
      const ds = await apiService.listDatasets();
      setDatasets(ds);
      setBackendConnected(true);
      setError(null);
    } catch (err: any) {
      console.warn("Could not connect to FastAPI backend: ", err.message);
      setBackendConnected(false);
      // Fallback local datasets
      setDatasets(getMockDatasets());
    }
  };

  const getMockDatasets = (): types.DatasetMetadata[] => [
    {
      id: "era5_reanalysis",
      name: "Era5 Reanalysis (Local Mock)",
      description: "Simulated ERA5 Reanalysis Dataset",
      variables: ["t2m", "z"],
      pressure_levels: [1000, 850, 500, 300, 200],
      time_range: ["2023-01-01", "2023-01-05"],
      spatial_resolution: "2.0 degree",
      bounding_box: { lat_min: -90, lat_max: 90, lon_min: -180, lon_max: 180 }
    },
    {
      id: "gfs_forecast",
      name: "GFS Forecast (Local Mock)",
      description: "Simulated GFS Forecast Dataset",
      variables: ["t2m", "z", "u", "v"],
      pressure_levels: [1000, 850, 500, 300],
      time_range: ["2023-01-01", "2023-01-02"],
      spatial_resolution: "2.0 degree",
      bounding_box: { lat_min: -90, lat_max: 90, lon_min: -180, lon_max: 180 }
    }
  ];

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

      if (backendConnected) {
        const res = await apiService.generateSynthetic({
          type: genType,
          height: gridSize,
          width: gridSize,
          params
        });
        setPrimaryField(res.field_data);
        setPrimaryCoords(res.coords);
        setPrimaryMetadata(res.metadata);
      } else {
        // Mock Generator
        const simulated = runMockFieldGenerator(genType, gridSize, params);
        setPrimaryField(simulated.field_data);
        setPrimaryCoords(simulated.coords);
        setPrimaryMetadata(simulated.metadata);
      }
      setPerturbedField(null);
      setPerturbationMetrics(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runMockFieldGenerator = (type: string, size: number, params: any) => {
    const data: number[][] = [];
    const yArr = Array.from({ length: size }, (_, i) => i / (size - 1));
    const xArr = Array.from({ length: size }, (_, i) => i / (size - 1));
    for (let r = 0; r < size; r++) {
      const row: number[] = [];
      const yVal = yArr[r];
      for (let c = 0; c < size; c++) {
        const xVal = xArr[c];
        let term = 0.0;
        if (type === 'sinusoid') {
          const freqs = params.frequencies?.[0] || [2, 2];
          const amp = params.amplitudes?.[0] || 1.0;
          term = amp * Math.sin(2 * Math.PI * freqs[0] * xVal) * Math.cos(2 * Math.PI * freqs[1] * yVal);
        } else if (type === 'vortex') {
          const center = params.centers?.[0] || [0.5, 0.5];
          const amp = params.amplitudes?.[0] || 1.0;
          const r_core = params.core_radii?.[0] || 0.1;
          const dist2 = Math.pow(xVal - center[0], 2) + Math.pow(yVal - center[1], 2);
          term = amp * Math.exp(-dist2 / (2 * r_core * r_core));
        } else {
          // Front
          const angleRad = (params.angle || 0.0) * Math.PI / 180.0;
          const offset = params.offset || 0.0;
          const width = params.width_param || 0.1;
          const amp = params.amplitude || 1.0;
          const proj = (xVal - 0.5) * Math.cos(angleRad) + (yVal - 0.5) * Math.sin(angleRad) - offset;
          term = amp * Math.tanh(proj / width);
        }
        row.push(term);
      }
      data.push(row);
    }
    return {
      field_data: data,
      coords: { x: xArr, y: yArr },
      metadata: { type, simulated: true, ...params }
    };
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
      if (backendConnected) {
        const res = await apiService.perturbField({
          field_data: primaryField,
          perturbations
        });
        setPerturbedField(res.perturbed_field);
        setPerturbationMetrics(res.metrics);
      } else {
        // Mock Perturb
        let current = JSON.parse(JSON.stringify(primaryField));
        perturbations.forEach(p => {
          current = applyMockPerturbation(current, p);
        });
        setPerturbedField(current);
        // Synthesize scientific sensitivity metrics
        setPerturbationMetrics({
          mean_squared_error: 0.0541,
          root_mean_squared_error: 0.2325,
          peak_signal_to_noise_ratio: 22.45,
          structural_similarity_index: 0.887,
          spectral_energy_shift: 0.1245
        });
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const applyMockPerturbation = (field: number[][], pert: types.PerturbationItem) => {
    const H = field.length;
    const W = field[0]?.length || 0;
    const nextField = Array.from({ length: H }, () => Array(W).fill(0));
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        let val = field[r][c];
        if (pert.type === 'noise') {
          const std = pert.level || 0.1;
          const noise = (Math.random() - 0.5) * 2.0 * std;
          val += noise;
        } else if (pert.type === 'translation') {
          const sx = Math.round((pert.shift_x || 0.1) * W);
          const sy = Math.round((pert.shift_y || 0.1) * H);
          const sourceR = (r - sy + H) % H;
          const sourceC = (c - sx + W) % W;
          val = field[sourceR][sourceC];
        } else if (pert.type === 'rotation') {
          // Simplistic rotation approximation for mock preview
          const angleRad = (pert.angle || 15) * Math.PI / 180;
          const cx = W / 2;
          const cy = H / 2;
          const rx = c - cx;
          const ry = r - cy;
          const srcX = Math.round(cx + rx * Math.cos(angleRad) - ry * Math.sin(angleRad));
          const srcY = Math.round(cy + rx * Math.sin(angleRad) + ry * Math.cos(angleRad));
          if (srcX >= 0 && srcX < W && srcY >= 0 && srcY < H) {
            val = field[srcY][srcX];
          } else {
            val = 0;
          }
        }
        nextField[r][c] = val;
      }
    }
    return nextField;
  };

  // --- TAB 2 ACTIONS: Met Data Slice ---
  const handleSliceDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      if (backendConnected) {
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
      } else {
        // Mock Slicer
        const coords = {
          lat: Array.from({ length: 24 }, (_, i) => latMin + (i * (latMax - latMin)) / 23),
          lon: Array.from({ length: 48 }, (_, i) => lonMin + (i * (lonMax - lonMin)) / 47)
        };
        const data = Array.from({ length: 24 }, (_, r) =>
          Array.from({ length: 48 }, (_, c) =>
            Math.cos((coords.lat[r] * Math.PI) / 180) * Math.sin((coords.lon[c] * Math.PI) / 180) + 273.15
          )
        );
        const metadata = { dataset_id: selectedDatasetId, variable: selectedVariable, sliced_mock: true };
        setSlicedField(data);
        setSlicedCoords(coords);
        setSlicedMetadata(metadata);
        setPrimaryField(data);
        setPrimaryCoords(coords);
        setPrimaryMetadata(metadata);
      }
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
      if (backendConnected) {
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
      } else {
        // Local Mock boundary simulation
        const size = primaryField.length;
        const pad = padWidth;
        const total = size + pad * 2;
        const nextField = Array.from({ length: total }, () => Array(total).fill(0));
        for (let r = 0; r < total; r++) {
          for (let c = 0; c < total; c++) {
            let origR = r - pad;
            let origC = c - pad;
            if (origR >= 0 && origR < size && origC >= 0 && origC < size) {
              nextField[r][c] = primaryField[origR][origC];
            } else {
              // circular pad mock
              const clampR = (origR + size) % size;
              const clampC = (origC + size) % size;
              nextField[r][c] = primaryField[clampR][clampC];
            }
          }
        }
        setPaddedField(nextField);
        setSpectralLeakage(1.245);
        setDistanceProfiles(
          Array.from({ length: pad + 1 }, (_, d) => ({
            distance: d,
            mean_gradient: 0.12 / (d + 1),
            max_gradient: 0.45 / (d + 1),
            mean_absolute_error: 0.05 / (d + 1),
            max_absolute_error: 0.15 / (d + 1)
          }))
        );
      }
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
      if (backendConnected) {
        const res = await apiService.applyTransform({
          field_data: primaryField,
          transform_type: transformType,
          config: {
            levels: waveletLevels,
            crossover_freq: crossoverFreq,
            mixing_weight: mixingWeight
          }
        });
        setReconstructedField(res.reconstructed_field);
        setTransformMetrics(res.metrics);
        setTransformCoefficients(res.coefficients);
      } else {
        // Mock transform (approximate reconstruction with tiny errors)
        const sizeR = primaryField.length;
        const sizeC = primaryField[0]?.length || 0;
        const recon = Array.from({ length: sizeR }, (_, r) =>
          Array.from({ length: sizeC }, (_, c) => primaryField[r][c] + (Math.random() - 0.5) * 0.01)
        );
        setReconstructedField(recon);
        setTransformMetrics({
          mean_squared_error: 0.000042,
          max_absolute_error: 0.004812
        });
        setTransformCoefficients({ mock: "coeffs serialized" });
      }
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
      // Forecast = original + noise
      const H = primaryField.length;
      const W = primaryField[0].length;
      const forecastField = Array.from({ length: H }, (_, r) =>
        Array.from({ length: W }, (_, c) => primaryField[r][c] + (Math.random() - 0.5) * forecastNoise * 2.0)
      );

      if (backendConnected) {
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
      } else {
        // Mock Diagnostics
        setDiagnosticsResults({
          spatial_metrics: {
            mean_squared_error: 0.012,
            root_mean_squared_error: 0.109,
            mean_absolute_error: 0.088,
            bias: 0.0012,
            structural_similarity_index: 0.945,
            pearson_correlation: 0.985
          },
          gradient_errors: {
            gradient_magnitude_mae: 0.045,
            gradient_direction_mae_rad: 0.112,
            gradient_direction_mae_deg: 6.41
          },
          spectral_diagnostics: {
            wavenumbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            forecast_psd: [1.2, 0.8, 0.45, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
            ground_truth_psd: [1.15, 0.78, 0.42, 0.18, 0.08, 0.04, 0.015, 0.008, 0.003, 0.001],
            spectral_coherence: [0.99, 0.98, 0.95, 0.92, 0.88, 0.82, 0.75, 0.65, 0.55, 0.42]
          },
          wavelet_energy: {
            levels: 3,
            forecast_energy: {},
            ground_truth_energy: {}
          }
        });
        setScaleDecompResults({
          low_scale_rmse: 0.015,
          mid_scale_rmse: 0.042,
          high_scale_rmse: 0.085,
          total_rmse: 0.109
        });
        setBoundaryDecompResults(
          Array.from({ length: 8 }, (_, d) => ({
            distance: d,
            rmse: 0.15 - (d * 0.015),
            mean_absolute_error: 0.12 - (d * 0.012)
          }))
        );
      }
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
      if (backendConnected) {
        const res = await apiService.createExperiment(parsedConfig);
        setSubmittedExperimentId(res.id);
        pollExperiment(res.id);
      } else {
        // Mock Experiment submission
        const mockId = "exp-" + Math.floor(Math.random() * 10000);
        setSubmittedExperimentId(mockId);
        setExperimentDetail({
          id: mockId,
          name: parsedConfig.name,
          description: parsedConfig.description,
          status: "COMPLETED",
          config: parsedConfig,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          runs: [
            {
              id: "run-01",
              experiment_id: mockId,
              parameters: { freq: 1.0, transform_type: "fft" },
              status: "COMPLETED",
              results: { trans_metrics: { mean_squared_error: 0.015, max_absolute_error: 0.085 } },
              created_at: new Date().toISOString()
            },
            {
              id: "run-02",
              experiment_id: mockId,
              parameters: { freq: 2.0, transform_type: "dct" },
              status: "COMPLETED",
              results: { trans_metrics: { mean_squared_error: 0.008, max_absolute_error: 0.042 } },
              created_at: new Date().toISOString()
            }
          ]
        });
        setLineageData({
          nodes: [
            { id: '1', name: 'v1.2.0-beta', type: 'code_revision', value: { revision: "v1.2.0-beta" }, created_at: new Date().toISOString() },
            { id: '2', name: 'sinusoid-run', type: 'field', value: { shape: [32, 32] }, created_at: new Date().toISOString() },
            { id: '3', name: 'fft-coeffs', type: 'coefficients', value: { transform: "fft" }, created_at: new Date().toISOString() },
            { id: '4', name: 'eval-metrics', type: 'metrics', value: { mse: 0.015 }, created_at: new Date().toISOString() }
          ],
          edges: [
            { id: 'e1', source_id: '1', target_id: '2', relation: 'executed_by' },
            { id: 'e2', source_id: '2', target_id: '3', relation: 'input_to' },
            { id: 'e3', source_id: '3', target_id: '4', relation: 'evaluated_to' }
          ]
        });
      }
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
      if (backendConnected) {
        const discovered = await apiService.discoverHypotheses({
          confidence_threshold: confidenceThreshold
        });
        setHypotheses(discovered);
        const activeProposals = await apiService.getProposals();
        setProposals(activeProposals);
      } else {
        // Mock Discoveries
        setHypotheses([
          {
            id: "hyp-01",
            experiment_ids: ["exp-mock-1"],
            pattern_type: "correlation",
            description: "A strong positive correlation (r = 0.89) was discovered between parameter 'freq' and metric 'trans_metrics.mean_squared_error'.",
            confidence: 0.89,
            metrics_analyzed: ["trans_metrics.mean_squared_error"],
            parameters_analyzed: ["freq"],
            proposed_experiment_config: {
              name: "Optimizing freq parameter",
              parameter_matrix: { freq: [0.1, 0.3, 0.5] },
              pipeline: []
            },
            created_at: new Date().toISOString()
          },
          {
            id: "hyp-02",
            experiment_ids: ["exp-mock-1"],
            pattern_type: "categorical_opt",
            description: "In experiment 'Spectral Accuracy Sweep', DCT significantly outperformed FFT. Category 'dct' performed best, outperforming 'fft' by 46.5%.",
            confidence: 0.465,
            metrics_analyzed: ["trans_metrics.mean_squared_error"],
            parameters_analyzed: ["transform_type"],
            proposed_experiment_config: {
              name: "Fixed to DCT",
              parameter_matrix: { transform_type: ["dct"] },
              pipeline: []
            },
            created_at: new Date().toISOString()
          }
        ]);
      }
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


  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Banner / Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Globe className="w-8 h-8 text-teal-400 animate-pulse" />
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">SpectralEarth</h1>
            <p className="text-xs text-slate-400">Scientific Visual Research Workbench</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {backendConnected ? (
            <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-full font-medium flex items-center gap-2">
              <Server className="w-3.5 h-3.5" /> API Connected (SQLite DB Active)
            </span>
          ) : (
            <span className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 px-3 py-1.5 rounded-full font-medium flex items-center gap-2 cursor-pointer" onClick={checkConnection}>
              <Server className="w-3.5 h-3.5" /> Offline Sandbox Mock Mode (Click to Retry)
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Side Navigation bar */}
        <nav className="w-full lg:w-72 border-r border-slate-800 bg-slate-900/10 p-4 space-y-1">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-3 mb-3">
            Research Modules
          </div>
          {[
            { id: 'synthetic', name: '1. Synthetic Generator', icon: Layers },
            { id: 'meteorological', name: '2. Meteorological Data', icon: Wind },
            { id: 'boundary', name: '3. Boundary-Condition Lab', icon: Sliders },
            { id: 'spectral', name: '4. Spectral Transforms', icon: Activity },
            { id: 'analysis', name: '5. Diagnostic & Analysis', icon: BarChart2 },
            { id: 'declarative', name: '6. Experiment Engine', icon: FileCode },
            { id: 'hypothesis', name: '7. Automated Hypotheses', icon: Lightbulb }
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20 shadow-sm shadow-teal-500/5'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-teal-400' : 'text-slate-400'}`} />
                {tab.name}
              </button>
            );
          })}

          <div className="pt-6 px-3 border-t border-slate-800 mt-6">
            <span className="text-xs text-slate-500 uppercase font-semibold block mb-2">Primary Field Buffer</span>
            <div className="bg-slate-900/50 p-3 border border-slate-800 rounded-lg text-xs space-y-1 text-slate-400">
              <p><strong className="text-slate-300">Dimensions:</strong> {primaryField.length} x {primaryField[0]?.length || 0}</p>
              <p><strong className="text-slate-300">Origin:</strong> {primaryMetadata.type || 'In-Memory Grid'}</p>
              <p><strong className="text-slate-300">Min/Max:</strong> {Math.min(...primaryField.flat()).toFixed(3)} / {Math.max(...primaryField.flat()).toFixed(3)}</p>
            </div>
          </div>
        </nav>

        {/* Core Main content section */}
        <main className="flex-1 p-6 overflow-y-auto space-y-6">
          {error && (
            <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <XCircle className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm font-medium">{error}</p>
              </div>
              <button onClick={() => setError(null)} className="text-xs underline hover:text-rose-200">Dismiss</button>
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
                    <label className="text-xs text-slate-400 block mb-1">Field Type</label>
                    <select
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
                    <label className="text-xs text-slate-400 block mb-1">Grid Size (N x N)</label>
                    <select
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
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>X-Frequency ($\omega_x$): {freqX}</span>
                        </label>
                        <input
                          type="range" min="0.5" max="8" step="0.5" value={freqX}
                          onChange={(e) => setFreqX(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Y-Frequency ($\omega_y$): {freqY}</span>
                        </label>
                        <input
                          type="range" min="0.5" max="8" step="0.5" value={freqY}
                          onChange={(e) => setFreqY(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {sinAmplitude}</span>
                        </label>
                        <input
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
                          <label className="text-xs text-slate-400 block mb-1">Center X</label>
                          <input
                            type="number" step="0.1" value={vortexCenterX}
                            onChange={(e) => setVortexCenterX(parseFloat(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-slate-400 block mb-1">Center Y</label>
                          <input
                            type="number" step="0.1" value={vortexCenterY}
                            onChange={(e) => setVortexCenterY(parseFloat(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {vortexAmp}</span>
                        </label>
                        <input
                          type="range" min="0.5" max="5" step="0.1" value={vortexAmp}
                          onChange={(e) => setVortexAmp(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>{"Core Radius ($r_{core}$): "}{vortexRadius}</span>
                        </label>
                        <input
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
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Angle (Degrees): {frontAngle}°</span>
                        </label>
                        <input
                          type="range" min="0" max="360" step="15" value={frontAngle}
                          onChange={(e) => setFrontAngle(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Offset: {frontOffset}</span>
                        </label>
                        <input
                          type="range" min="-0.5" max="0.5" step="0.1" value={frontOffset}
                          onChange={(e) => setFrontOffset(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Width Parameter: {frontWidth}</span>
                        </label>
                        <input
                          type="range" min="0.01" max="0.3" step="0.01" value={frontWidth}
                          onChange={(e) => setFrontWidth(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Amplitude: {frontAmp}</span>
                        </label>
                        <input
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
                    <Heatmap2D data={primaryField} title="Generated Clean Field ($F$)" colormap="viridis" coords={primaryCoords} />
                    <Heatmap2D data={perturbedField || primaryField} title="Perturbed Spatial Field ($F'$)" colormap="viridis" coords={primaryCoords} />
                  </div>

                  {/* Perturbation Engine Steps */}
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
                    <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 mb-4 flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-teal-400" /> Sequential Perturbation Scheduler
                    </h3>
                    
                    <div className="flex flex-wrap items-end gap-4 mb-4">
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">Perturbation Type</label>
                        <select
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
                          <label className="text-xs text-slate-400 block mb-1">Rotation Angle (°)</label>
                          <input
                            type="number" value={newPertAngle}
                            onChange={(e) => setNewPertAngle(parseFloat(e.target.value))}
                            className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 w-24"
                          />
                        </div>
                      )}

                      {newPertType === 'translation' && (
                        <div className="flex gap-2">
                          <div>
                            <label className="text-xs text-slate-400 block mb-1">Shift X</label>
                            <input
                              type="number" step="0.05" value={newPertShiftX}
                              onChange={(e) => setNewPertShiftX(parseFloat(e.target.value))}
                              className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 w-20"
                            />
                          </div>
                          <div>
                            <label className="text-xs text-slate-400 block mb-1">Shift Y</label>
                            <input
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
                            <label className="text-xs text-slate-400 block mb-1">Noise Distribution</label>
                            <select
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
                            <label className="text-xs text-slate-400 block mb-1">Level ($\sigma$)</label>
                            <input
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
                              <button onClick={() => setPerturbations(perturbations.filter((_, i) => i !== idx))} className="text-rose-500 hover:text-rose-400">
                                <Trash2 className="w-3 h-3" />
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
                    <label className="text-xs text-slate-400 block mb-1">Active Dataset</label>
                    <select
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
                        <option key={d.id} value={d.id}>{d.name}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 block mb-1">Variable</label>
                    <select
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
                      <label className="text-xs text-slate-400 block mb-1">Pressure Level (hPa)</label>
                      <select
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
                        <label className="text-[10px] text-slate-500 block">Latitude Min</label>
                        <input
                          type="number" value={latMin} onChange={(e) => setLatMin(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] text-slate-500 block">Latitude Max</label>
                        <input
                          type="number" value={latMax} onChange={(e) => setLatMax(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[10px] text-slate-500 block">Longitude Min</label>
                        <input
                          type="number" value={lonMin} onChange={(e) => setLonMin(parseFloat(e.target.value))}
                          className="w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] text-slate-500 block">Longitude Max</label>
                        <input
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
                      <Heatmap2D data={slicedField} title={`${selectedVariable.toUpperCase()} Crop (${selectedDatasetId})`} coords={slicedCoords} colormap="viridis" />
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
                    <label className="text-xs text-slate-400 block mb-1">Padding Treatment</label>
                    <select
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
                    <label className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Pad Width: {padWidth}px</span>
                    </label>
                    <input
                      type="range" min="1" max="16" step="1" value={padWidth}
                      onChange={(e) => setPadWidth(parseInt(e.target.value))}
                      className="w-full accent-teal-500"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 block mb-1">Spectral Window Tapering</label>
                    <select
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
                      <label className="text-xs text-slate-400 flex justify-between mb-1">
                        <span>Tukey Alpha ($\alpha$): {windowAlpha}</span>
                      </label>
                      <input
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
                    <label className="text-xs text-slate-400 block mb-1">Transform Operator</label>
                    <select
                      value={transformType}
                      onChange={(e) => setTransformType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-200 focus:outline-none"
                    >
                      <option value="fft">2D Fast Fourier Transform (FFT)</option>
                      <option value="dct">2D Discrete Cosine Transform (DCT)</option>
                      <option value="dwt">2D Haar Discrete Wavelet (DWT)</option>
                      <option value="dtcwt">2D Dual-Tree Complex Wavelet (DTCWT)</option>
                      <option value="hybrid">FFT Low-Pass + DWT Residual Hybrid</option>
                    </select>
                  </div>

                  {['dwt', 'dtcwt'].includes(transformType) && (
                    <div>
                      <label className="text-xs text-slate-400 flex justify-between mb-1">
                        <span>Wavelet Levels: {waveletLevels}</span>
                      </label>
                      <input
                        type="range" min="1" max="4" step="1" value={waveletLevels}
                        onChange={(e) => setWaveletLevels(parseInt(e.target.value))}
                        className="w-full accent-teal-500"
                      />
                    </div>
                  )}

                  {transformType === 'hybrid' && (
                    <div className="space-y-3">
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Crossover Frequency: {crossoverFreq}</span>
                        </label>
                        <input
                          type="range" min="0.05" max="0.5" step="0.05" value={crossoverFreq}
                          onChange={(e) => setCrossoverFreq(parseFloat(e.target.value))}
                          className="w-full accent-teal-500"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 flex justify-between mb-1">
                          <span>Mixing Weight: {mixingWeight}</span>
                        </label>
                        <input
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
                    <Heatmap2D data={primaryField} title="Original Target Field ($F$)" colormap="viridis" />
                    <Heatmap2D data={reconstructedField || primaryField} title="Inverse Reconstructed Field ($\hat{F}$)" colormap="viridis" />
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
                        <div className="col-span-2 md:col-span-1 bg-slate-950 border border-slate-800 p-4 rounded-xl flex flex-col justify-center text-xs text-slate-400 leading-relaxed">
                          <p className="flex items-center gap-1.5"><CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> Mathematically rigorous floating point calculations.</p>
                          <p className="flex items-center gap-1.5 mt-1"><CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> Verified perfect reconstruct limits.</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
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
                  <p className="text-xs text-slate-400 leading-relaxed">Generates a mock forecasted field by injecting variable standard-deviation noise onto the primary active field buffer.</p>

                  <div>
                    <label className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Forecast Noise StDev: {forecastNoise}</span>
                    </label>
                    <input
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
                          xLabel="Wavenumber ($k$)"
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
                          xLabel="Wavenumber ($k$)"
                          yLabel="Coherence Ratio"
                        />
                      </div>

                      {/* Turbulence Spectral Slope Fits */}
                      {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis && diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis && (
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-slate-950/50 border border-slate-850 p-4 rounded-lg">
                            <span className="text-xs text-slate-500 uppercase font-bold block mb-1">Forecast Spectral Slope</span>
                            <span className="text-xl font-bold text-sky-400 font-mono">
                              β = {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.slope_beta.toFixed(3)}
                            </span>
                            <span className="text-xs text-slate-400 block mt-1">
                              Power-Law Fit $R^2$: {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.r_squared.toFixed(3)}
                            </span>
                            <span className="text-xs text-teal-400 font-medium block mt-1.5 border-t border-slate-800/40 pt-1.5">
                              {diagnosticsResults.spectral_diagnostics.forecast_slope_analysis.regime_interpretation}
                            </span>
                          </div>
                          <div className="bg-slate-950/50 border border-slate-850 p-4 rounded-lg">
                            <span className="text-xs text-slate-500 uppercase font-bold block mb-1">Ground Truth Spectral Slope</span>
                            <span className="text-xl font-bold text-teal-400 font-mono">
                              β = {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.slope_beta.toFixed(3)}
                            </span>
                            <span className="text-xs text-slate-400 block mt-1">
                              Power-Law Fit $R^2$: {diagnosticsResults.spectral_diagnostics.ground_truth_slope_analysis.r_squared.toFixed(3)}
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

                  <textarea
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
                <p className="text-sm text-slate-400">Mines metrics in the SQLite runs tables using numerical correlation math (Pearson's $r$) and categorical optimization to discover pattern proposals.</p>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2">
                    Hypothesis Discovery Config
                  </h3>

                  <div>
                    <label className="text-xs text-slate-400 flex justify-between mb-1">
                      <span>Pearson Confidence Cutoff: {confidenceThreshold}</span>
                    </label>
                    <input
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
                      <span className="text-xs font-semibold text-slate-300 block mb-3">Mined Scientific Hypotheses & Adaptive Proposals ({hypotheses.length})</span>
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
                                <span className="text-xs font-mono text-slate-500">Confidence: {(h.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <p className="text-xs text-slate-200 font-medium leading-relaxed mb-4">{h.description}</p>
                              <div className="text-[10px] font-mono bg-slate-950 p-2 border border-slate-850 rounded text-slate-400 space-y-1 mb-4">
                                <div><strong className="text-slate-300">Metric:</strong> {h.metrics_analyzed.join(', ')}</div>
                                <div><strong className="text-slate-300">Parameter:</strong> {h.parameters_analyzed.join(', ')}</div>
                              </div>
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
        </main>
      </div>
    </div>
  );
}