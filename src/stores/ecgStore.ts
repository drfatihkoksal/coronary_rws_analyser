/**
 * ECG Store
 *
 * Manages ECG waveform data and R-peak detection for cardiac sync.
 * R-peaks define beat boundaries for RWS calculation.
 */

import { create } from 'zustand';
import type { ECGData, BeatBoundary } from '../types';
import { dicomApi } from '../lib/api';

interface SelectedBeat {
  startFrame: number;
  endFrame: number;
  beatNumber: number | null;
}

interface ECGState {
  // Loading state
  isLoading: boolean;
  isLoaded: boolean;
  hasEcg: boolean;
  error: string | null;

  // ECG data
  ecgData: ECGData | null;

  // Derived data
  beatBoundaries: BeatBoundary[];
  heartRate: number | null;

  // Selected beat for RWS calculation
  selectedBeat: SelectedBeat | null;

  // Actions
  loadEcg: () => Promise<void>;
  getBeatForFrame: (frameIndex: number, frameRate: number) => number | null;
  getFrameRangeForBeat: (beatIndex: number) => { start: number; end: number } | null;
  setSelectedBeat: (beat: SelectedBeat | null) => void;
  reset: () => void;
}

export const useEcgStore = create<ECGState>()((set, get) => ({
  // Initial state
  isLoading: false,
  isLoaded: false,
  hasEcg: false,
  error: null,
  ecgData: null,
  beatBoundaries: [],
  heartRate: null,
  selectedBeat: null,

  // Load ECG from current DICOM
  loadEcg: async () => {
    set({ isLoading: true, error: null });

    try {
      const ecgData = await dicomApi.getEcg();

      // Calculate beat boundaries from R-peaks
      const beatBoundaries: BeatBoundary[] = [];

      if (ecgData.rPeaks && ecgData.rPeaks.length >= 2) {
        for (let i = 0; i < ecgData.rPeaks.length - 1; i++) {
          beatBoundaries.push({
            startFrame: ecgData.rPeaks[i],  // Sample index (needs conversion)
            endFrame: ecgData.rPeaks[i + 1],
            rPeakSample: ecgData.rPeaks[i],
          });
        }
      }

      set({
        isLoading: false,
        isLoaded: true,
        hasEcg: true,
        ecgData,
        beatBoundaries,
        heartRate: ecgData.heartRate,
      });

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load ECG';
      set({
        isLoading: false,
        isLoaded: false,
        hasEcg: false,
        error: message,
        ecgData: null,
        beatBoundaries: [],
        heartRate: null,
      });
    }
  },

  // Get beat index for a frame
  getBeatForFrame: (frameIndex, frameRate) => {
    const { ecgData, beatBoundaries } = get();

    if (!ecgData || beatBoundaries.length === 0) {
      return null;
    }

    // Convert frame index to ECG sample index
    // Assume ECG and video are synchronized
    const timeSeconds = frameIndex / frameRate;
    const sampleIndex = Math.floor(timeSeconds * ecgData.samplingRate);

    // Find which beat this sample belongs to
    for (let i = 0; i < beatBoundaries.length; i++) {
      const beat = beatBoundaries[i];
      if (sampleIndex >= beat.startFrame && sampleIndex < beat.endFrame) {
        return i;
      }
    }

    return null;
  },

  // Get frame range for a beat
  getFrameRangeForBeat: (beatIndex) => {
    const { beatBoundaries, ecgData } = get();

    if (!ecgData || beatIndex < 0 || beatIndex >= beatBoundaries.length) {
      return null;
    }

    const beat = beatBoundaries[beatIndex];

    // Note: These are sample indices, need conversion to frame indices
    // The conversion depends on frame rate which should be provided by caller
    return {
      start: beat.startFrame,
      end: beat.endFrame,
    };
  },

  // Set selected beat for RWS calculation
  setSelectedBeat: (beat) => set({ selectedBeat: beat }),

  // Reset
  reset: () => {
    set({
      isLoading: false,
      isLoaded: false,
      hasEcg: false,
      error: null,
      ecgData: null,
      beatBoundaries: [],
      heartRate: null,
      selectedBeat: null,
    });
  },
}));

// Selectors
export const selectEcgSignal = (state: ECGState) => state.ecgData?.signal ?? [];
export const selectRPeaks = (state: ECGState) => state.ecgData?.rPeaks ?? [];
export const selectHeartRate = (state: ECGState) => state.heartRate;
export const selectBeatBoundaries = (state: ECGState) => state.beatBoundaries;
