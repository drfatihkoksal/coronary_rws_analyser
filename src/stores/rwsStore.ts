/**
 * RWS Store (Primary Feature)
 *
 * Manages Radial Wall Strain calculations and results.
 * RWS = (Dmax - Dmin) / Dmax × 100%
 *
 * Clinical thresholds:
 * - Normal: < 8%
 * - Intermediate: 8-12%
 * - Elevated: > 12%
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { RWSResult, RWSSummary, RWSInterpretation } from '../types';
import { rwsApi, OutlierMethod } from '../lib/api';

interface RWSState {
  // Loading state
  isCalculating: boolean;
  error: string | null;

  // Results (per-beat)
  results: RWSResult[];

  // Current/Latest result
  currentResult: RWSResult | null;

  // Summary statistics
  summary: RWSSummary | null;

  // Settings
  outlierMethod: OutlierMethod;

  // Actions
  calculate: (startFrame: number, endFrame: number, options?: { beatNumber?: number; outlierMethod?: OutlierMethod }) => Promise<RWSResult | null>;
  calculateFromDiameters: (
    mldDiameters: number[],
    proximalDiameters: number[],
    distalDiameters: number[],
    beatNumber?: number
  ) => Promise<RWSResult | null>;
  loadResults: () => Promise<void>;
  loadSummary: () => Promise<void>;

  // Interpretation
  interpret: (rwsValue: number) => RWSInterpretation;

  // Getters
  getResultForBeat: (beatNumber: number) => RWSResult | undefined;
  getMldRwsValues: () => number[];

  // Setters
  setIsCalculating: (v: boolean) => void;
  setCurrentResult: (r: RWSResult | null) => void;
  setOutlierMethod: (m: OutlierMethod) => void;

  // Clear
  clearResults: () => Promise<void>;
  reset: () => void;
}

export const useRwsStore = create<RWSState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    isCalculating: false,
    error: null,
    results: [],
    currentResult: null,
    summary: null,
    outlierMethod: 'hampel' as OutlierMethod,

    // Calculate RWS for frame range
    calculate: async (startFrame, endFrame, options = {}) => {
      set({ isCalculating: true, error: null });

      try {
        const { outlierMethod: storeOutlierMethod } = get();
        const beatNumber = options.beatNumber;
        const outlierMethod = options.outlierMethod ?? storeOutlierMethod;
        const result = await rwsApi.calculate(startFrame, endFrame, { beatNumber, outlierMethod });

        set((state) => ({
          isCalculating: false,
          currentResult: result,
          results: [...state.results, result],
        }));

        return result;

      } catch (error) {
        const message = error instanceof Error ? error.message : 'RWS calculation failed';
        set({ isCalculating: false, error: message });
        return null;
      }
    },

    // Calculate from diameter arrays
    calculateFromDiameters: async (mldDiameters, proximalDiameters, distalDiameters, beatNumber) => {
      set({ isCalculating: true, error: null });

      try {
        const result = await rwsApi.calculateFromDiameters(
          mldDiameters,
          proximalDiameters,
          distalDiameters,
          undefined,
          beatNumber
        );

        set((state) => ({
          isCalculating: false,
          currentResult: result,
          results: [...state.results, result],
        }));

        return result;

      } catch (error) {
        const message = error instanceof Error ? error.message : 'RWS calculation failed';
        set({ isCalculating: false, error: message });
        return null;
      }
    },

    // Load all results from backend
    loadResults: async () => {
      try {
        const response = await rwsApi.getResults();
        set({ results: response.results });
      } catch (error) {
        console.error('Failed to load RWS results:', error);
      }
    },

    // Load summary
    loadSummary: async () => {
      try {
        const summary = await rwsApi.getSummary();
        set({ summary });
      } catch (error) {
        console.error('Failed to load RWS summary:', error);
      }
    },

    // Local interpretation (no API call)
    interpret: (rwsValue) => {
      if (rwsValue < 8) return 'normal';
      if (rwsValue < 12) return 'intermediate';
      return 'elevated';
    },

    // Get result for specific beat
    getResultForBeat: (beatNumber) => {
      return get().results.find(r => r.beatNumber === beatNumber);
    },

    // Get all MLD RWS values
    getMldRwsValues: () => {
      return get().results.map(r => r.mldRws.rws);
    },

    // Setters
    setIsCalculating: (v) => set({ isCalculating: v }),
    setCurrentResult: (r) => set({ currentResult: r }),
    setOutlierMethod: (m) => set({ outlierMethod: m }),

    // Clear results
    clearResults: async () => {
      try {
        await rwsApi.clear();
        set({ results: [], currentResult: null, summary: null });
      } catch (error) {
        console.error('Failed to clear RWS results:', error);
      }
    },

    // Reset
    reset: () => {
      set({
        isCalculating: false,
        error: null,
        results: [],
        currentResult: null,
        summary: null,
        outlierMethod: 'hampel' as OutlierMethod,
      });
    },
  }))
);

// Selectors
export const selectCurrentResult = (state: RWSState) => state.currentResult;
export const selectResults = (state: RWSState) => state.results;
export const selectSummary = (state: RWSState) => state.summary;
export const selectIsCalculating = (state: RWSState) => state.isCalculating;

// Helper: Get clinical interpretation color
export function getRwsColor(interpretation: RWSInterpretation): string {
  switch (interpretation) {
    case 'normal':
      return '#22c55e'; // green-500
    case 'intermediate':
      return '#f59e0b'; // amber-500
    case 'elevated':
      return '#ef4444'; // red-500
    default:
      return '#6b7280'; // gray-500
  }
}

// Helper: Get clinical interpretation label
export function getRwsLabel(interpretation: RWSInterpretation): string {
  switch (interpretation) {
    case 'normal':
      return 'Normal';
    case 'intermediate':
      return 'Intermediate';
    case 'elevated':
      return 'Elevated Risk';
    default:
      return 'Unknown';
  }
}
