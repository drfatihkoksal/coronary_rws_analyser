/**
 * TypeScript Type Definitions
 *
 * All types for the Coronary RWS Analyser application.
 */

// ============================================================================
// DICOM Types
// ============================================================================

export interface DicomMetadata {
  patientId: string;
  patientName: string;
  studyDate: string | null;
  studyTime: string | null;
  studyDescription: string | null;
  modality: string;
  manufacturer: string | null;
  rows: number;
  columns: number;
  numFrames: number;
  bitsStored: number;
  pixelSpacing: [number, number] | null;
  pixelSpacingSource: string | null;
  frameRate: number;
  frameTime: number | null;
  primaryAngle: number | null;
  secondaryAngle: number | null;
  angleLabel: string | null;
  hasEcg: boolean;
  ecgType: string | null;
}

export interface FrameData {
  frameIndex: number;
  data: string; // Base64 encoded PNG
  width: number;
  height: number;
}

// ============================================================================
// ECG Types
// ============================================================================

export interface ECGData {
  signal: number[];
  time: number[];
  rPeaks: number[] | null;
  samplingRate: number;
  heartRate: number | null;
  duration: number;
  metadata: Record<string, unknown>;
}

export interface BeatBoundary {
  startFrame: number;
  endFrame: number;
  rPeakSample: number;
}

// ============================================================================
// Geometry Types
// ============================================================================

export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ============================================================================
// Segmentation Types
// ============================================================================

export type SegmentationEngine = 'nnunet' | 'angiopy';

export interface SegmentationEngineInfo {
  available: boolean;
  name: string;
  description: string;
  requires: string;
  seedPoints: boolean;
  reference?: string;
}

export interface SegmentationResult {
  frameIndex: number;
  mask: Uint8Array | null;
  probabilityMap: Float32Array | null;
  centerline: Point[];
  width: number;
  height: number;
  engineUsed?: SegmentationEngine;
  numSeedPoints?: number;
}

// ============================================================================
// Tracking Types
// ============================================================================

export type TrackingMethod = 'csrt';

export interface TrackingResult {
  success: boolean;
  frameIndex: number;
  roi: BoundingBox | null;
  confidence: number;
  method: TrackingMethod;
  errorMessage: string | null;
}

export interface TrackingState {
  isInitialized: boolean;
  isTracking: boolean;
  currentFrame: number;
  initialRoi: BoundingBox | null;
}

export interface PropagationProgress {
  isRunning: boolean;
  currentFrame: number;
  totalFrames: number;
  progress: number; // 0-100
  status: 'idle' | 'running' | 'completed' | 'stopped' | 'failed';
}

// ============================================================================
// QCA Types
// ============================================================================

export interface QCAMetrics {
  frameIndex: number;
  diameterProfile: number[]; // mm
  mld: number; // Minimum Lumen Diameter (mm)
  mldIndex: number;
  mldPosition: Point;
  proximalRd: number; // Proximal Reference Diameter (mm)
  proximalRdIndex: number;
  proximalRdPosition: Point;
  distalRd: number; // Distal Reference Diameter (mm)
  distalRdIndex: number;
  distalRdPosition: Point;
  interpolatedRd: number;
  diameterStenosis: number; // DS%
  lesionLength: number | null; // mm
  pixelSpacing: number;
  numPoints: number;
}

// ============================================================================
// RWS Types (Primary Feature)
// ============================================================================

export type RWSInterpretation = 'normal' | 'intermediate' | 'elevated';

export interface RWSMeasurement {
  position: 'mld' | 'proximal' | 'distal';
  dmax: number; // mm
  dmaxFrame: number;
  dmin: number; // mm
  dminFrame: number;
  rws: number; // Percentage
  interpretation: RWSInterpretation;
}

export interface RWSResult {
  mldRws: RWSMeasurement;
  proximalRws: RWSMeasurement;
  distalRws: RWSMeasurement;
  beatNumber: number | null;
  startFrame: number;
  endFrame: number;
  numFrames: number;
  averageRws: number;
}

export interface RWSSummary {
  numBeats: number;
  mldRwsMean: number;
  mldRwsStd: number;
  mldRwsMin: number;
  mldRwsMax: number;
  proximalRwsMean: number;
  distalRwsMean: number;
}

// ============================================================================
// Calibration Types
// ============================================================================

export type CalibrationSource = 'dicom' | 'manual_catheter' | 'manual';

export interface Calibration {
  pixelSpacing: [number, number]; // [row, col] mm/pixel
  source: CalibrationSource;
  catheterSizeFr: number | null;
}

// ============================================================================
// Player Types
// ============================================================================

export type PlaybackState = 'stopped' | 'playing' | 'paused';

export interface ViewTransform {
  scale: number;
  x: number;
  y: number;
}

// ============================================================================
// Export Types
// ============================================================================

export type ExportFormat = 'csv' | 'json' | 'pdf';

export interface ExportOptions {
  format: ExportFormat;
  includeQca: boolean;
  includeRws: boolean;
  includeMetadata: boolean;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiError {
  detail: string;
  errorCode?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  service: string;
  version?: string;
}
