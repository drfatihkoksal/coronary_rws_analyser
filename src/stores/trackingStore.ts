/**
 * Tracking Store
 *
 * Manages CSRT ROI tracking state across frames.
 * Uses Map for per-frame tracking results.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { BoundingBox, TrackingResult, PropagationProgress, RoiMode } from '../types';
import { trackingApi } from '../lib/api';

interface TrackingState {
  // Tracking state
  isInitialized: boolean;
  isTracking: boolean;
  isPropagating: boolean;
  error: string | null;

  // ROI mode setting
  roiMode: RoiMode;

  // Current tracking data
  currentRoi: BoundingBox | null;
  currentConfidence: number;

  // Per-frame results (Map-based)
  trackingResults: Map<number, TrackingResult>;

  // Propagation progress
  propagationProgress: PropagationProgress;

  // Initial state (for reset)
  initialFrame: number | null;
  initialRoi: BoundingBox | null;

  // Actions
  setRoiMode: (mode: RoiMode) => void;
  initialize: (frameIndex: number, roi: BoundingBox) => Promise<boolean>;
  trackFrame: (frameIndex: number) => Promise<TrackingResult | null>;
  propagate: (startFrame: number, endFrame: number, direction?: 'forward' | 'backward') => Promise<void>;
  stopPropagation: () => void;

  // Getters
  getResult: (frameIndex: number) => TrackingResult | undefined;

  // Update
  updateCurrentState: (roi: BoundingBox, confidence: number) => void;
  setCurrentRoi: (roi: BoundingBox | null) => void;
  setIsPropagating: (isPropagating: boolean) => void;
  setPropagationProgress: (progress: Partial<PropagationProgress>) => void;

  // Reset
  reset: () => void;
}

const DEFAULT_PROPAGATION_PROGRESS: PropagationProgress = {
  isRunning: false,
  currentFrame: 0,
  totalFrames: 0,
  progress: 0,
  status: 'idle',
};

export const useTrackingStore = create<TrackingState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    isInitialized: false,
    isTracking: false,
    isPropagating: false,
    error: null,
    roiMode: 'fixed_150x150',
    currentRoi: null,
    currentConfidence: 0,
    trackingResults: new Map(),
    propagationProgress: { ...DEFAULT_PROPAGATION_PROGRESS },
    initialFrame: null,
    initialRoi: null,

    // Set ROI mode
    setRoiMode: (mode) => {
      set({ roiMode: mode });
    },

    // Initialize CSRT tracker (ROI only)
    initialize: async (frameIndex, roi) => {
      const { roiMode } = get();
      set({ isTracking: true, error: null });

      try {
        const response = await trackingApi.initialize({ frameIndex, roi, roiMode });

        // Use actual ROI returned by backend (may be adjusted by roi_mode)
        const actualRoi = response.roi || roi;

        set({
          isTracking: false,
          isInitialized: true,
          currentRoi: actualRoi,
          currentConfidence: 1.0,
          initialFrame: frameIndex,
          initialRoi: actualRoi,
          trackingResults: new Map(),
        });

        return true;

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Initialization failed';
        set({ isTracking: false, error: message });
        return false;
      }
    },

    // Track single frame
    trackFrame: async (frameIndex) => {
      const { isInitialized } = get();

      if (!isInitialized) {
        set({ error: 'Tracker not initialized' });
        return null;
      }

      set({ isTracking: true, error: null });

      try {
        const result = await trackingApi.track(frameIndex);

        // Store result
        set((state) => {
          const newResults = new Map(state.trackingResults);
          newResults.set(frameIndex, result);

          return {
            isTracking: false,
            trackingResults: newResults,
            currentRoi: result.roi ?? state.currentRoi,
            currentConfidence: result.confidence,
          };
        });

        return result;

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Tracking failed';
        set({ isTracking: false, error: message });
        return null;
      }
    },

    // Propagate tracking
    propagate: async (startFrame, endFrame, direction = 'forward') => {
      const { isInitialized } = get();

      if (!isInitialized) {
        set({ error: 'Tracker not initialized' });
        return;
      }

      set({
        isPropagating: true,
        error: null,
        propagationProgress: {
          isRunning: true,
          currentFrame: startFrame,
          totalFrames: Math.abs(endFrame - startFrame) + 1,
          progress: 0,
          status: 'running',
        },
      });

      try {
        const response = await trackingApi.propagate(startFrame, endFrame, direction);

        // Store all results
        set((state) => {
          const newResults = new Map(state.trackingResults);

          for (const result of response.results) {
            newResults.set(result.frameIndex, result);
          }

          // Update current state with last successful result
          const lastResult = response.results[response.results.length - 1];

          return {
            isPropagating: false,
            trackingResults: newResults,
            currentRoi: lastResult?.roi ?? state.currentRoi,
            currentConfidence: lastResult?.confidence ?? state.currentConfidence,
            propagationProgress: {
              isRunning: false,
              currentFrame: endFrame,
              totalFrames: Math.abs(endFrame - startFrame) + 1,
              progress: 100,
              status: response.stoppedEarly ? 'stopped' : 'completed',
            },
          };
        });

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Propagation failed';
        set({
          isPropagating: false,
          error: message,
          propagationProgress: {
            ...get().propagationProgress,
            isRunning: false,
            status: 'failed',
          },
        });
      }
    },

    // Stop propagation
    stopPropagation: () => {
      set((state) => ({
        isPropagating: false,
        propagationProgress: {
          ...state.propagationProgress,
          isRunning: false,
          status: 'stopped',
        },
      }));
    },

    // Get result for frame
    getResult: (frameIndex) => get().trackingResults.get(frameIndex),

    // Update current state
    updateCurrentState: (roi, confidence) => {
      set({
        currentRoi: roi,
        currentConfidence: confidence,
      });
    },

    // Set current ROI (for annotation)
    setCurrentRoi: (roi) => {
      set({ currentRoi: roi });
    },

    // Set isPropagating (for WebSocket control)
    setIsPropagating: (isPropagating) => {
      set({ isPropagating });
    },

    // Set propagation progress (for WebSocket updates)
    setPropagationProgress: (progress) => {
      set((state) => ({
        propagationProgress: { ...state.propagationProgress, ...progress },
      }));
    },

    // Reset
    reset: async () => {
      try {
        await trackingApi.reset();
      } catch {
        // Ignore reset errors
      }

      // Keep roiMode, reset everything else
      const { roiMode } = get();
      set({
        isInitialized: false,
        isTracking: false,
        isPropagating: false,
        error: null,
        roiMode, // Preserve ROI mode setting
        currentRoi: null,
        currentConfidence: 0,
        trackingResults: new Map(),
        propagationProgress: { ...DEFAULT_PROPAGATION_PROGRESS },
        initialFrame: null,
        initialRoi: null,
      });
    },
  }))
);

// Selectors
export const selectCurrentRoi = (state: TrackingState) => state.currentRoi;
export const selectIsTracking = (state: TrackingState) => state.isTracking;
export const selectIsPropagating = (state: TrackingState) => state.isPropagating;
export const selectPropagationProgress = (state: TrackingState) => state.propagationProgress;
export const selectRoiMode = (state: TrackingState) => state.roiMode;
