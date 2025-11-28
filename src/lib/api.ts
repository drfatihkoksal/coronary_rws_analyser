/**
 * API Client for Coronary RWS Analyser
 *
 * Communicates with the Python FastAPI backend.
 */

import type {
  DicomMetadata,
  FrameData,
  ECGData,
  BoundingBox,
  Point,
  QCAMetrics,
  RWSResult,
  RWSSummary,
  TrackingResult,
  PropagationProgress,
  Calibration,
  HealthResponse,
  SegmentationEngine,
  SegmentationEngineInfo,
} from '../types';

// API Base URL
const API_BASE = 'http://127.0.0.1:8000';

// ============================================================================
// Utility Functions
// ============================================================================

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function toCamelCase(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

function convertKeys<T>(obj: unknown, converter: (s: string) => string): T {
  if (Array.isArray(obj)) {
    return obj.map(item => convertKeys(item, converter)) as T;
  }
  if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        converter(key),
        convertKeys(value, converter)
      ])
    ) as T;
  }
  return obj as T;
}

// ============================================================================
// DICOM API
// ============================================================================

export const dicomApi = {
  async load(file: File): Promise<DicomMetadata> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/dicom/load`, {
      method: 'POST',
      body: formData,
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<DicomMetadata>(data, toCamelCase);
  },

  async getMetadata(): Promise<DicomMetadata> {
    const response = await fetch(`${API_BASE}/dicom/metadata`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<DicomMetadata>(data, toCamelCase);
  },

  async getFrame(frameIndex: number): Promise<FrameData> {
    const response = await fetch(`${API_BASE}/dicom/frame/${frameIndex}`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<FrameData>(data, toCamelCase);
  },

  async getFrames(start: number, end: number): Promise<{ frames: FrameData[]; total: number }> {
    const response = await fetch(`${API_BASE}/dicom/frames?start=${start}&end=${end}`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async getEcg(): Promise<ECGData> {
    const response = await fetch(`${API_BASE}/dicom/ecg`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<ECGData>(data, toCamelCase);
  },

  async getNumFrames(): Promise<number> {
    const response = await fetch(`${API_BASE}/dicom/num_frames`);
    const data = await handleResponse<{ num_frames: number }>(response);
    return data.num_frames;
  },

  async clear(): Promise<void> {
    await fetch(`${API_BASE}/dicom/clear`, { method: 'POST' });
  },

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/dicom/health`);
    return handleResponse<HealthResponse>(response);
  },
};

// ============================================================================
// Segmentation API
// ============================================================================

export const segmentationApi = {
  /**
   * Segment vessel using selected engine
   * @param frameIndex Frame to segment
   * @param options.engine 'nnunet' (ROI-based) or 'angiopy' (seed-guided)
   * @param options.roi ROI for nnU-Net (x, y, w, h)
   * @param options.seedPoints Seed points for AngioPy (2-10 points)
   * @param options.returnProbability Include probability map
   */
  async segment(
    frameIndex: number,
    options: {
      engine?: SegmentationEngine;
      roi?: BoundingBox;
      seedPoints?: Point[];
      returnProbability?: boolean;
    } = {}
  ): Promise<{
    frameIndex: number;
    mask: string;
    probabilityMap: string | null;
    width: number;
    height: number;
    engineUsed: string;
    numSeedPoints?: number;
  }> {
    const {
      engine = 'nnunet',
      roi,
      seedPoints,
      returnProbability = true,
    } = options;

    const body: Record<string, unknown> = {
      frame_index: frameIndex,
      engine,
      return_probability: returnProbability,
    };

    // Add ROI for nnU-Net
    if (roi) {
      body.roi = [roi.x, roi.y, roi.width, roi.height];
    }

    // Add seed points for AngioPy
    if (seedPoints && seedPoints.length > 0) {
      body.seed_points = seedPoints.map(p => ({ x: p.x, y: p.y }));
    }

    const response = await fetch(`${API_BASE}/segmentation/segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async extractCenterline(
    frameIndex: number,
    maskBase64?: string,
    method: 'skeleton' | 'distance' | 'mcp' = 'skeleton',
    numSeeds = 3
  ): Promise<{
    frameIndex: number;
    centerline: Point[];
    seedPoints: Point[];
    numPoints: number;
  }> {
    const params = new URLSearchParams({
      frame_index: frameIndex.toString(),
      method,
      num_seeds: numSeeds.toString(),
    });

    if (maskBase64) {
      params.append('mask_base64', maskBase64);
    }

    const response = await fetch(`${API_BASE}/segmentation/centerline?${params}`, {
      method: 'POST',
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async segmentAndExtract(
    frameIndex: number,
    roi?: BoundingBox
  ): Promise<{
    frameIndex: number;
    mask: string;
    probabilityMap: string | null;
    centerline: Point[];
    seedPoints: Point[];
    maskShape: [number, number];
  }> {
    const params = new URLSearchParams({ frame_index: frameIndex.toString() });
    if (roi) {
      // Convert to integers (backend expects int)
      params.append('x', Math.round(roi.x).toString());
      params.append('y', Math.round(roi.y).toString());
      params.append('w', Math.round(roi.width).toString());
      params.append('h', Math.round(roi.height).toString());
    }

    const response = await fetch(`${API_BASE}/segmentation/segment-and-extract?${params}`, {
      method: 'POST',
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async getModelInfo(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_BASE}/segmentation/model-info`);
    return handleResponse(response);
  },

  /**
   * Get available segmentation engines and their status
   */
  async getEngines(): Promise<{
    defaultEngine: SegmentationEngine;
    engines: Record<SegmentationEngine, SegmentationEngineInfo>;
  }> {
    const response = await fetch(`${API_BASE}/segmentation/engines`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/segmentation/health`);
    return handleResponse<HealthResponse>(response);
  },

  // =========================================================================
  // YOLO Keypoint Detection
  // =========================================================================

  /**
   * Detect vessel keypoints using YOLO within ROI
   * Returns 5 keypoints: start, quarter, center, three_quarter, end
   */
  async detectKeypoints(
    frameIndex: number,
    roi: { x: number; y: number; width: number; height: number }
  ): Promise<{
    frameIndex: number;
    roi: [number, number, number, number];
    success: boolean;
    confidence: number;
    keypoints: Point[];
    numKeypoints: number;
    keypointNames: string[];
    error?: string;
  }> {
    const params = new URLSearchParams({
      frame_index: frameIndex.toString(),
      roi_x: Math.round(roi.x).toString(),
      roi_y: Math.round(roi.y).toString(),
      roi_w: Math.round(roi.width).toString(),
      roi_h: Math.round(roi.height).toString(),
    });

    const response = await fetch(`${API_BASE}/segmentation/detect-keypoints?${params}`, {
      method: 'POST',
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  /**
   * Automatic segmentation using YOLO keypoint detection + AngioPy
   * Pipeline: ROI → YOLO (5 keypoints) → AngioPy → Mask → MCP Centerline
   */
  async segmentAuto(
    frameIndex: number,
    roi: { x: number; y: number; width: number; height: number },
    returnProbability = false
  ): Promise<{
    frameIndex: number;
    roi: [number, number, number, number];
    success: boolean;
    mask: string;
    width: number;
    height: number;
    seedPoints: Point[];
    centerline: Point[];
    yoloConfidence: number;
    engineUsed: string;
    error?: string;
  }> {
    const params = new URLSearchParams({
      frame_index: frameIndex.toString(),
      roi_x: Math.round(roi.x).toString(),
      roi_y: Math.round(roi.y).toString(),
      roi_w: Math.round(roi.width).toString(),
      roi_h: Math.round(roi.height).toString(),
      return_probability: returnProbability.toString(),
    });

    const response = await fetch(`${API_BASE}/segmentation/segment-auto?${params}`, {
      method: 'POST',
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  /**
   * Complete automatic analysis: YOLO → AngioPy → Centerline
   */
  async segmentAutoWithCenterline(
    frameIndex: number,
    roi: { x: number; y: number; width: number; height: number }
  ): Promise<{
    frameIndex: number;
    roi: [number, number, number, number];
    success: boolean;
    mask: string;
    seedPoints: Point[];
    centerline: Point[];
    yoloConfidence: number;
    engineUsed: string;
    error?: string;
  }> {
    const params = new URLSearchParams({
      frame_index: frameIndex.toString(),
      roi_x: Math.round(roi.x).toString(),
      roi_y: Math.round(roi.y).toString(),
      roi_w: Math.round(roi.width).toString(),
      roi_h: Math.round(roi.height).toString(),
    });

    const response = await fetch(`${API_BASE}/segmentation/segment-auto-with-centerline?${params}`, {
      method: 'POST',
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },
};

// ============================================================================
// Tracking API
// ============================================================================

export const trackingApi = {
  /**
   * Initialize CSRT tracker for ROI tracking
   * @param params.frameIndex Frame to initialize on
   * @param params.roi ROI bounding box to track
   */
  async initialize(params: {
    frameIndex: number;
    roi: BoundingBox;
  }): Promise<{ status: string; frameIndex: number }> {
    const body = {
      frame_index: params.frameIndex,
      roi: {
        x: Math.round(params.roi.x),
        y: Math.round(params.roi.y),
        width: Math.round(params.roi.width),
        height: Math.round(params.roi.height),
      },
    };

    const response = await fetch(`${API_BASE}/tracking/initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async track(frameIndex: number): Promise<TrackingResult> {
    const body = { frame_index: frameIndex };

    const response = await fetch(`${API_BASE}/tracking/track`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<TrackingResult>(data, toCamelCase);
  },

  async propagate(
    startFrame: number,
    endFrame: number,
    direction: 'forward' | 'backward' = 'forward'
  ): Promise<{
    results: TrackingResult[];
    numFramesTracked: number;
    stoppedEarly: boolean;
    stopReason: string | null;
  }> {
    const body = {
      start_frame: startFrame,
      end_frame: endFrame,
      direction,
    };

    const response = await fetch(`${API_BASE}/tracking/propagate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async getStatus(): Promise<PropagationProgress> {
    const response = await fetch(`${API_BASE}/tracking/status`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<PropagationProgress>(data, toCamelCase);
  },

  async reset(): Promise<void> {
    await fetch(`${API_BASE}/tracking/reset`, { method: 'POST' });
  },

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/tracking/health`);
    return handleResponse<HealthResponse>(response);
  },
};

// ============================================================================
// QCA API
// ============================================================================

export type QCAMethod = 'gaussian' | 'parabolic' | 'threshold';

export const qcaApi = {
  async calculate(
    frameIndex: number,
    options: {
      numPoints?: number;
      method?: QCAMethod;
    } = {}
  ): Promise<QCAMetrics> {
    const { numPoints = 50, method = 'gaussian' } = options;

    const body = {
      frame_index: frameIndex,
      num_points: numPoints,
      method,
    };

    const response = await fetch(`${API_BASE}/qca/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<QCAMetrics>(data, toCamelCase);
  },

  async getMetrics(frameIndex: number): Promise<QCAMetrics> {
    const response = await fetch(`${API_BASE}/qca/metrics/${frameIndex}`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<QCAMetrics>(data, toCamelCase);
  },

  async getDiameterAtPosition(
    frameIndex: number,
    y: number,
    x: number
  ): Promise<{ frameIndex: number; position: Point; diameterMm: number }> {
    const params = new URLSearchParams({
      frame_index: frameIndex.toString(),
      y: y.toString(),
      x: x.toString(),
    });

    const response = await fetch(`${API_BASE}/qca/diameter-at-position?${params}`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/qca/health`);
    return handleResponse<HealthResponse>(response);
  },
};

// ============================================================================
// RWS API (Primary Feature)
// ============================================================================

export type OutlierMethod = 'none' | 'hampel' | 'double_hampel' | 'iqr' | 'temporal';

export const rwsApi = {
  async calculate(
    startFrame: number,
    endFrame: number,
    options: {
      beatNumber?: number;
      outlierMethod?: OutlierMethod;
    } = {}
  ): Promise<RWSResult> {
    const { beatNumber, outlierMethod = 'hampel' } = options;

    const body = {
      start_frame: startFrame,
      end_frame: endFrame,
      beat_number: beatNumber ?? null,
      outlier_method: outlierMethod,
    };

    const response = await fetch(`${API_BASE}/rws/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<RWSResult>(data, toCamelCase);
  },

  async calculateFromDiameters(
    mldDiameters: number[],
    proximalDiameters: number[],
    distalDiameters: number[],
    frameIndices?: number[],
    beatNumber?: number
  ): Promise<RWSResult> {
    const body = {
      mld_diameters: mldDiameters,
      proximal_diameters: proximalDiameters,
      distal_diameters: distalDiameters,
      frame_indices: frameIndices ?? null,
      beat_number: beatNumber ?? null,
    };

    const response = await fetch(`${API_BASE}/rws/calculate-from-diameters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<RWSResult>(data, toCamelCase);
  },

  async getResults(): Promise<{ results: RWSResult[]; count: number }> {
    const response = await fetch(`${API_BASE}/rws/results`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async getSummary(): Promise<RWSSummary> {
    const response = await fetch(`${API_BASE}/rws/summary`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys<RWSSummary>(data, toCamelCase);
  },

  async interpret(rwsValue: number): Promise<{
    rws: number;
    interpretation: string;
    description: string;
    riskLevel: string;
  }> {
    const response = await fetch(`${API_BASE}/rws/interpret/${rwsValue}`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async clear(): Promise<void> {
    await fetch(`${API_BASE}/rws/clear`, { method: 'POST' });
  },

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/rws/health`);
    return handleResponse<HealthResponse>(response);
  },
};

// ============================================================================
// Calibration API
// ============================================================================

export const calibrationApi = {
  async getCurrent(): Promise<{
    pixelSpacing: number;
    source: string;
    confidence: string;
    catheterSize?: string;
  }> {
    const response = await fetch(`${API_BASE}/calibration/current`);
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async fromDicom(): Promise<{
    pixelSpacing: number;
    source: string;
    confidence: string;
  }> {
    const response = await fetch(`${API_BASE}/calibration/from-dicom`, {
      method: 'POST',
    });
    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async manual(params: {
    catheterSize: string;
    customSizeMm?: number;
    seedPoints: number[][];
    frameIndex: number;
    method?: string;
    nPoints?: number;
  }): Promise<{
    success: boolean;
    newPixelSpacing?: number;
    oldPixelSpacing?: number;
    catheterSize?: string;
    knownDiameterMm?: number;
    measuredDiameterPx?: number;
    qualityScore?: number;
    qualityNotes?: string[];
    errorMessage?: string;
  }> {
    const body = {
      catheter_size: params.catheterSize,
      custom_size_mm: params.customSizeMm,
      seed_points: params.seedPoints,
      frame_index: params.frameIndex,
      method: params.method || 'gaussian',
      n_points: params.nPoints || 50,
    };

    const response = await fetch(`${API_BASE}/calibration/manual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async getCatheterSizes(): Promise<Record<string, number>> {
    const response = await fetch(`${API_BASE}/calibration/catheter-sizes`);
    return handleResponse(response);
  },
};

// ============================================================================
// Export API
// ============================================================================

export const exportApi = {
  async qca(params: {
    format: string;
    frameIndices?: number[];
    includeDiameterProfiles?: boolean;
  }): Promise<{
    success: boolean;
    filename?: string;
    path?: string;
    format?: string;
    sizeBytes?: number;
    errorMessage?: string;
  }> {
    const body = {
      format: params.format,
      frame_indices: params.frameIndices,
      include_diameter_profiles: params.includeDiameterProfiles ?? true,
    };

    const response = await fetch(`${API_BASE}/export/qca`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async rws(params: {
    format: string;
    beatIndices?: number[];
    includeTemporalData?: boolean;
  }): Promise<{
    success: boolean;
    filename?: string;
    path?: string;
    format?: string;
    sizeBytes?: number;
    errorMessage?: string;
  }> {
    const body = {
      format: params.format,
      beat_indices: params.beatIndices,
      include_temporal_data: params.includeTemporalData ?? true,
    };

    const response = await fetch(`${API_BASE}/export/rws`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  async all(params: {
    format: string;
    includeQca?: boolean;
    includeRws?: boolean;
    includeMetadata?: boolean;
  }): Promise<{
    success: boolean;
    filename?: string;
    path?: string;
    format?: string;
    sizeBytes?: number;
    errorMessage?: string;
  }> {
    const body = {
      format: params.format,
      include_qca: params.includeQca ?? true,
      include_rws: params.includeRws ?? true,
      include_metadata: params.includeMetadata ?? true,
    };

    const response = await fetch(`${API_BASE}/export/all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await handleResponse<Record<string, unknown>>(response);
    return convertKeys(data, toCamelCase);
  },

  getDownloadUrl(filename: string): string {
    return `${API_BASE}/export/download/${filename}`;
  },
};

// ============================================================================
// Health Check
// ============================================================================

export async function checkBackendHealth(): Promise<{
  overall: boolean;
  services: Record<string, HealthResponse>;
}> {
  const services: Record<string, HealthResponse> = {};
  let overall = true;

  try {
    services.dicom = await dicomApi.health();
  } catch {
    services.dicom = { status: 'unhealthy', service: 'dicom' };
    overall = false;
  }

  try {
    services.segmentation = await segmentationApi.health();
  } catch {
    services.segmentation = { status: 'unhealthy', service: 'segmentation' };
    overall = false;
  }

  try {
    services.tracking = await trackingApi.health();
  } catch {
    services.tracking = { status: 'unhealthy', service: 'tracking' };
    overall = false;
  }

  try {
    services.qca = await qcaApi.health();
  } catch {
    services.qca = { status: 'unhealthy', service: 'qca' };
    overall = false;
  }

  try {
    services.rws = await rwsApi.health();
  } catch {
    services.rws = { status: 'unhealthy', service: 'rws' };
    overall = false;
  }

  return { overall, services };
}

// ============================================================================
// Unified API Export
// ============================================================================

export const api = {
  dicom: dicomApi,
  segmentation: segmentationApi,
  tracking: trackingApi,
  qca: qcaApi,
  rws: rwsApi,
  calibration: calibrationApi,
  export: exportApi,
  checkHealth: checkBackendHealth,
};
