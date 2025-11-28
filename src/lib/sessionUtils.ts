/**
 * Session Utilities
 *
 * Centralized session management for resetting all stores
 * when a new DICOM series is loaded.
 */

import { usePlayerStore } from '../stores/playerStore';
import { useDicomStore } from '../stores/dicomStore';
import { useSegmentationStore } from '../stores/segmentationStore';
import { useAnnotationStore } from '../stores/annotationStore';
import { useQcaStore } from '../stores/qcaStore';
import { useRwsStore } from '../stores/rwsStore';
import { useEcgStore } from '../stores/ecgStore';
import { useTrackingStore } from '../stores/trackingStore';
import { useCalibrationStore } from '../stores/calibrationStore';

/**
 * Reset all analysis stores when loading a new DICOM series.
 * This should be called BEFORE loading a new file.
 *
 * Preserves:
 * - Calibration settings (user may want to keep manual calibration)
 *
 * Clears:
 * - All segmentation masks and centerlines
 * - All annotations (seed points, ROI)
 * - All QCA metrics
 * - All RWS results
 * - All tracking results
 * - ECG data
 * - Player state (frame, playback)
 */
export async function resetAllAnalysis(): Promise<void> {
  console.log('[Session] Resetting all analysis stores...');

  // Reset in dependency order (most dependent first)

  // 1. RWS depends on QCA
  useRwsStore.getState().reset();

  // 2. QCA depends on segmentation
  useQcaStore.getState().clearAll();

  // 3. Tracking depends on annotations (async - may call backend)
  await useTrackingStore.getState().reset();

  // 4. Segmentation
  useSegmentationStore.getState().reset();

  // 5. Annotations
  useAnnotationStore.getState().reset();

  // 6. ECG
  useEcgStore.getState().reset();

  // 7. Player (frame position, playback state)
  usePlayerStore.getState().reset();

  // Note: We DON'T reset calibrationStore - user may want to keep their calibration

  console.log('[Session] All analysis stores reset');
}

/**
 * Reset everything including DICOM data.
 * Use this for a complete fresh start.
 */
export async function resetAll(): Promise<void> {
  await resetAllAnalysis();

  // Also reset DICOM store
  useDicomStore.getState().reset();

  // Optionally reset calibration (uncomment if needed)
  // useCalibrationStore.getState().reset();

  console.log('[Session] Complete reset done');
}

/**
 * Get summary of current session state.
 * Useful for debugging and display.
 */
export function getSessionSummary(): SessionSummary {
  const segStore = useSegmentationStore.getState();
  const annStore = useAnnotationStore.getState();
  const qcaStore = useQcaStore.getState();
  const rwsStore = useRwsStore.getState();
  const trackStore = useTrackingStore.getState();
  const calStore = useCalibrationStore.getState();

  // Count frames with mask and centerline from frameData Map
  let framesWithMask = 0;
  let framesWithCenterline = 0;
  segStore.frameData.forEach((data) => {
    if (data.mask) framesWithMask++;
    if (data.centerline && data.centerline.length > 0) framesWithCenterline++;
  });

  return {
    framesWithMask,
    framesWithCenterline,
    framesWithAnnotations: annStore.frameAnnotations.size,
    framesWithQCA: qcaStore.frameMetrics.size,
    rwsResultCount: rwsStore.results.length,
    framesTracked: trackStore.trackingResults.size,
    hasEcg: useEcgStore.getState().hasEcg,
    isCalibrated: calStore.calibration !== null,
  };
}

export interface SessionSummary {
  framesWithMask: number;
  framesWithCenterline: number;
  framesWithAnnotations: number;
  framesWithQCA: number;
  rwsResultCount: number;
  framesTracked: number;
  hasEcg: boolean;
  isCalibrated: boolean;
}

/**
 * Check if there's unsaved analysis data.
 * Can be used to warn user before loading new file.
 */
export function hasUnsavedAnalysis(): boolean {
  const summary = getSessionSummary();

  return (
    summary.framesWithMask > 0 ||
    summary.framesWithCenterline > 0 ||
    summary.framesWithQCA > 0 ||
    summary.rwsResultCount > 0
  );
}
