/**
 * QCA Store
 *
 * Map-based storage for per-frame QCA metrics.
 * Used by RWS calculation for diameter tracking.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { QCAMetrics } from '../types';
import { qcaApi, QCAMethod } from '../lib/api';

interface QCAState {
  // Loading state
  isCalculating: boolean;
  error: string | null;

  // Per-frame metrics (Map-based)
  frameMetrics: Map<number, QCAMetrics>;

  // Current frame metrics (for quick access)
  currentMetrics: QCAMetrics | null;

  // Analysis settings
  numPoints: number; // N-point analysis (30, 50, 70)
  method: QCAMethod; // Diameter calculation method

  // Actions
  calculateFrame: (frameIndex: number, options?: { numPoints?: number; method?: QCAMethod }) => Promise<QCAMetrics | null>;
  calculateRange: (startFrame: number, endFrame: number, options?: { numPoints?: number; method?: QCAMethod }) => Promise<void>;

  // Getters
  getMetrics: (frameIndex: number) => QCAMetrics | undefined;
  getDiameterProfile: (frameIndex: number) => number[] | undefined;
  getMldForRange: (startFrame: number, endFrame: number) => number[];

  // Setters
  setMetrics: (frameIndex: number, metrics: QCAMetrics) => void;
  setNumPoints: (n: number) => void;
  setMethod: (m: QCAMethod) => void;
  setIsCalculating: (v: boolean) => void;
  setCurrentMetrics: (m: QCAMetrics | null) => void;

  // Clear
  clearFrame: (frameIndex: number) => void;
  clearAll: () => void;
  reset: () => void;
}

export const useQcaStore = create<QCAState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    isCalculating: false,
    error: null,
    frameMetrics: new Map(),
    currentMetrics: null,
    numPoints: 50,
    method: 'gaussian' as QCAMethod,

    // Calculate QCA for single frame
    calculateFrame: async (frameIndex, options = {}) => {
      set({ isCalculating: true, error: null });

      try {
        const { numPoints: storeNumPoints, method: storeMethod } = get();
        const numPoints = options.numPoints ?? storeNumPoints;
        const method = options.method ?? storeMethod;
        const metrics = await qcaApi.calculate(frameIndex, { numPoints, method });

        // Store metrics
        set((state) => {
          const newMetrics = new Map(state.frameMetrics);
          newMetrics.set(frameIndex, metrics);
          return {
            isCalculating: false,
            frameMetrics: newMetrics,
            currentMetrics: metrics,
          };
        });

        return metrics;

      } catch (error) {
        const message = error instanceof Error ? error.message : 'QCA calculation failed';
        set({ isCalculating: false, error: message });
        return null;
      }
    },

    // Calculate QCA for frame range
    calculateRange: async (startFrame, endFrame, options = {}) => {
      set({ isCalculating: true, error: null });

      const { numPoints: storeNumPoints, method: storeMethod } = get();
      const numPoints = options.numPoints ?? storeNumPoints;
      const method = options.method ?? storeMethod;
      const errors: string[] = [];

      for (let i = startFrame; i <= endFrame; i++) {
        try {
          const metrics = await qcaApi.calculate(i, { numPoints, method });

          set((state) => {
            const newMetrics = new Map(state.frameMetrics);
            newMetrics.set(i, metrics);
            return { frameMetrics: newMetrics };
          });

        } catch (error) {
          const message = error instanceof Error ? error.message : `Frame ${i} failed`;
          errors.push(message);
        }
      }

      set({
        isCalculating: false,
        error: errors.length > 0 ? errors.join(', ') : null,
      });
    },

    // Getters
    getMetrics: (frameIndex) => get().frameMetrics.get(frameIndex),

    getDiameterProfile: (frameIndex) => {
      const metrics = get().frameMetrics.get(frameIndex);
      return metrics?.diameterProfile;
    },

    getMldForRange: (startFrame, endFrame) => {
      const { frameMetrics } = get();
      const mldValues: number[] = [];

      for (let i = startFrame; i <= endFrame; i++) {
        const metrics = frameMetrics.get(i);
        if (metrics) {
          mldValues.push(metrics.mld);
        }
      }

      return mldValues;
    },

    // Setters
    setMetrics: (frameIndex, metrics) => {
      set((state) => {
        const newMetrics = new Map(state.frameMetrics);
        newMetrics.set(frameIndex, metrics);
        return { frameMetrics: newMetrics };
      });
    },

    setNumPoints: (n) => set({ numPoints: n }),
    setMethod: (m) => set({ method: m }),
    setIsCalculating: (v) => set({ isCalculating: v }),
    setCurrentMetrics: (m) => set({ currentMetrics: m }),

    // Clear
    clearFrame: (frameIndex) => {
      set((state) => {
        const newMetrics = new Map(state.frameMetrics);
        newMetrics.delete(frameIndex);
        return { frameMetrics: newMetrics };
      });
    },

    clearAll: () => {
      set({
        frameMetrics: new Map(),
        currentMetrics: null,
      });
    },

    reset: () => {
      set({
        isCalculating: false,
        error: null,
        frameMetrics: new Map(),
        currentMetrics: null,
        numPoints: 50,
        method: 'gaussian' as QCAMethod,
      });
    },
  }))
);

// Selectors
export const selectCurrentMetrics = (state: QCAState) => state.currentMetrics;
export const selectIsCalculating = (state: QCAState) => state.isCalculating;
export const selectFrameMetricsCount = (state: QCAState) => state.frameMetrics.size;

// Helper to get metrics as object for RWS
export const getMetricsForRws = (state: QCAState) => {
  const result: Record<number, QCAMetrics> = {};
  state.frameMetrics.forEach((metrics, frameIndex) => {
    result[frameIndex] = metrics;
  });
  return result;
};
