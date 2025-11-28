/**
 * Zustand Stores Index
 *
 * Export all stores for the Coronary RWS Analyser application.
 *
 * Architecture:
 * - playerStore: Video playback, current frame, zoom/pan
 * - dicomStore: DICOM metadata, frame caching
 * - ecgStore: ECG data, R-peaks, beat boundaries
 * - segmentationStore: Masks, centerlines (Map-based)
 * - trackingStore: ROI tracking, propagation (Map-based)
 * - qcaStore: QCA metrics per frame (Map-based)
 * - rwsStore: RWS results (PRIMARY FEATURE)
 * - calibrationStore: Pixel-to-mm calibration
 */

// Player
export {
  usePlayerStore,
  selectCurrentFrame,
  selectPlaybackState,
  selectViewTransform,
  selectViewTransformVersion,
  selectAnnotationMode,
  type AnnotationMode,
} from './playerStore';

// DICOM
export {
  useDicomStore,
  selectMetadata,
  selectIsLoaded,
  selectPixelSpacing,
} from './dicomStore';

// ECG
export {
  useEcgStore,
  selectEcgSignal,
  selectRPeaks,
  selectHeartRate,
  selectBeatBoundaries,
} from './ecgStore';

// Segmentation
export {
  useSegmentationStore,
  selectIsSegmenting,
  selectSelectedEngine,
  selectSeedPoints,
  selectAvailableEngines,
  selectFrameData,
} from './segmentationStore';

// Tracking
export {
  useTrackingStore,
  selectCurrentRoi,
  selectIsTracking,
  selectIsPropagating,
  selectPropagationProgress,
} from './trackingStore';

// QCA
export {
  useQcaStore,
  selectCurrentMetrics,
  selectIsCalculating as selectIsQcaCalculating,
  selectFrameMetricsCount,
  getMetricsForRws,
} from './qcaStore';

// RWS (Primary Feature)
export {
  useRwsStore,
  selectCurrentResult,
  selectResults,
  selectSummary,
  selectIsCalculating as selectIsRwsCalculating,
  getRwsColor,
  getRwsLabel,
} from './rwsStore';

// Calibration
export {
  useCalibrationStore,
  selectCalibration,
  selectSource,
  selectIsCalibrated,
  CATHETER_SIZES,
} from './calibrationStore';

// Annotation
export {
  useAnnotationStore,
  selectSeedMode,
  selectROIMode,
  selectFrameAnnotations,
} from './annotationStore';

// Export
export {
  useExportStore,
  selectIsExporting,
  selectExportProgress,
  selectExportError,
  selectExportHistory,
} from './exportStore';

// Overlay Visibility
export {
  useOverlayStore,
  selectOverlayVisibility,
  selectMaskVisible,
  selectCenterlineVisible,
  selectSeedPointsVisible,
  selectYoloKeypointsVisible,
  selectRoiVisible,
} from './overlayStore';

// Type re-exports
export type { PlaybackState, ViewTransform } from '../types';
