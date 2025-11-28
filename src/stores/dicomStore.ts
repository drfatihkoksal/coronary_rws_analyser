/**
 * DICOM Store
 *
 * Manages DICOM metadata and frame caching.
 * Uses Map for frame storage to support sparse frame loading.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { DicomMetadata, FrameData } from '../types';
import { dicomApi } from '../lib/api';
import { usePlayerStore } from './playerStore';
import { resetAllAnalysis } from '../lib/sessionUtils';

interface DicomState {
  // Loading state
  isLoading: boolean;
  isLoaded: boolean;
  error: string | null;

  // Metadata
  metadata: DicomMetadata | null;

  // Frame cache (Map-based for sparse storage)
  frames: Map<number, FrameData>;

  // Actions
  loadFile: (file: File) => Promise<void>;
  getFrame: (frameIndex: number) => Promise<FrameData | null>;
  preloadFrames: (start: number, end: number) => Promise<void>;
  clearFrame: (frameIndex: number) => void;
  clearAllFrames: () => void;
  reset: () => void;
}

export const useDicomStore = create<DicomState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    isLoading: false,
    isLoaded: false,
    error: null,
    metadata: null,
    frames: new Map(),

    // Load DICOM file
    loadFile: async (file) => {
      set({ isLoading: true, error: null });

      try {
        // Reset all analysis data before loading new file
        await resetAllAnalysis();

        const metadata = await dicomApi.load(file);

        set({
          isLoading: false,
          isLoaded: true,
          metadata,
          frames: new Map(), // Clear existing frames
        });

        // Update player store with frame info
        usePlayerStore.getState().setTotalFrames(metadata.numFrames);
        usePlayerStore.getState().setFrameRate(metadata.frameRate);
        usePlayerStore.getState().setCurrentFrame(0);

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load DICOM';
        set({
          isLoading: false,
          isLoaded: false,
          error: message,
          metadata: null,
        });
        throw error;
      }
    },

    // Get frame (from cache or API)
    getFrame: async (frameIndex) => {
      const { frames, metadata } = get();

      // Check bounds
      if (!metadata || frameIndex < 0 || frameIndex >= metadata.numFrames) {
        return null;
      }

      // Return from cache if available
      const cached = frames.get(frameIndex);
      if (cached) {
        return cached;
      }

      // Fetch from API
      try {
        const frameData = await dicomApi.getFrame(frameIndex);

        // Update cache
        set((state) => {
          const newFrames = new Map(state.frames);
          newFrames.set(frameIndex, frameData);
          return { frames: newFrames };
        });

        return frameData;
      } catch (error) {
        console.error(`Failed to load frame ${frameIndex}:`, error);
        return null;
      }
    },

    // Preload frame range
    preloadFrames: async (start, end) => {
      const { metadata, frames } = get();
      if (!metadata) return;

      const clampedStart = Math.max(0, start);
      const clampedEnd = Math.min(metadata.numFrames - 1, end);

      const promises: Promise<void>[] = [];

      for (let i = clampedStart; i <= clampedEnd; i++) {
        if (!frames.has(i)) {
          promises.push(
            (async () => {
              try {
                const frameData = await dicomApi.getFrame(i);
                set((state) => {
                  const newFrames = new Map(state.frames);
                  newFrames.set(i, frameData);
                  return { frames: newFrames };
                });
              } catch (error) {
                console.error(`Failed to preload frame ${i}:`, error);
              }
            })()
          );
        }
      }

      await Promise.all(promises);
    },

    // Clear single frame from cache
    clearFrame: (frameIndex) => {
      set((state) => {
        const newFrames = new Map(state.frames);
        newFrames.delete(frameIndex);
        return { frames: newFrames };
      });
    },

    // Clear all frames
    clearAllFrames: () => {
      set({ frames: new Map() });
    },

    // Reset store
    reset: () => {
      set({
        isLoading: false,
        isLoaded: false,
        error: null,
        metadata: null,
        frames: new Map(),
      });

      // Also reset player
      usePlayerStore.getState().reset();
    },
  }))
);

// Selectors
export const selectMetadata = (state: DicomState) => state.metadata;
export const selectIsLoaded = (state: DicomState) => state.isLoaded;
export const selectPixelSpacing = (state: DicomState) => state.metadata?.pixelSpacing ?? null;
