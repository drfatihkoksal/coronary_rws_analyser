/**
 * Segmentation Store
 *
 * Map-based storage for per-frame segmentation results.
 * All overlays (mask, centerline, seed points) are cached per frame.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Point, BoundingBox, SegmentationResult, SegmentationEngine, SegmentationEngineInfo } from '../types';
import { segmentationApi } from '../lib/api';

// Per-frame cached data
interface FrameSegmentationData {
  mask: string | null;          // Base64 encoded mask
  probabilityMap: string | null; // Base64 encoded
  centerline: Point[];
  seedPoints: Point[];          // Generated seed points from centerline
}

interface SegmentationState {
  // Engine configuration
  selectedEngine: SegmentationEngine;
  availableEngines: Record<SegmentationEngine, SegmentationEngineInfo> | null;
  enginesLoaded: boolean;

  // Loading state
  isSegmenting: boolean;
  error: string | null;

  // Seed points for AngioPy (user-placed)
  seedPoints: Point[];
  maxSeedPoints: number;

  // Per-frame cached data (Map-based for sparse storage)
  frameData: Map<number, FrameSegmentationData>;

  // Engine actions
  setEngine: (engine: SegmentationEngine) => void;
  loadEngines: () => Promise<void>;

  // Seed point actions
  addSeedPoint: (point: Point) => void;
  removeSeedPoint: (index: number) => void;
  updateSeedPoint: (index: number, point: Point) => void;
  clearSeedPoints: () => void;
  setSeedPoints: (points: Point[]) => void;

  // Segmentation actions
  segmentFrame: (frameIndex: number, roi?: BoundingBox) => Promise<SegmentationResult | null>;
  extractCenterline: (frameIndex: number, method?: 'skeleton' | 'distance' | 'mcp') => Promise<Point[]>;
  segmentAndExtract: (frameIndex: number, roi?: BoundingBox) => Promise<void>;

  // Getters - return cached data for specific frame
  getFrameData: (frameIndex: number) => FrameSegmentationData | undefined;
  getMask: (frameIndex: number) => string | null;
  getCenterline: (frameIndex: number) => Point[];
  getSeedPoints: (frameIndex: number) => Point[];
  hasData: (frameIndex: number) => boolean;

  // Get all frames with data
  getFramesWithData: () => number[];

  // Clear
  clearFrame: (frameIndex: number) => void;
  clearAll: () => void;
  reset: () => void;
}

const EMPTY_FRAME_DATA: FrameSegmentationData = {
  mask: null,
  probabilityMap: null,
  centerline: [],
  seedPoints: [],
};

export const useSegmentationStore = create<SegmentationState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    selectedEngine: 'nnunet',
    availableEngines: null,
    enginesLoaded: false,
    isSegmenting: false,
    error: null,
    seedPoints: [],
    maxSeedPoints: 10,
    frameData: new Map(),

    // Set segmentation engine
    setEngine: (engine) => {
      set({ selectedEngine: engine });
    },

    // Load available engines from backend
    loadEngines: async () => {
      try {
        const result = await segmentationApi.getEngines();
        set({
          availableEngines: result.engines,
          selectedEngine: result.defaultEngine,
          enginesLoaded: true,
        });
      } catch (error) {
        console.error('Failed to load segmentation engines:', error);
        set({ enginesLoaded: true });
      }
    },

    // Add seed point
    addSeedPoint: (point) => {
      const { seedPoints, maxSeedPoints } = get();
      if (seedPoints.length >= maxSeedPoints) {
        console.warn(`Maximum ${maxSeedPoints} seed points allowed`);
        return;
      }
      set({ seedPoints: [...seedPoints, point] });
    },

    // Remove seed point by index
    removeSeedPoint: (index) => {
      const { seedPoints } = get();
      if (index < 0 || index >= seedPoints.length) return;
      const newPoints = [...seedPoints];
      newPoints.splice(index, 1);
      set({ seedPoints: newPoints });
    },

    // Update seed point position
    updateSeedPoint: (index, point) => {
      const { seedPoints } = get();
      if (index < 0 || index >= seedPoints.length) return;
      const newPoints = [...seedPoints];
      newPoints[index] = point;
      set({ seedPoints: newPoints });
    },

    // Clear all seed points
    clearSeedPoints: () => {
      set({ seedPoints: [] });
    },

    // Set all seed points at once
    setSeedPoints: (points) => {
      set({ seedPoints: points.slice(0, get().maxSeedPoints) });
    },

    // Segment a single frame
    segmentFrame: async (frameIndex, roi) => {
      const { selectedEngine, seedPoints } = get();
      set({ isSegmenting: true, error: null });

      try {
        if (selectedEngine === 'angiopy' && seedPoints.length < 2) {
          throw new Error('AngioPy requires at least 2 seed points');
        }

        const result = await segmentationApi.segment(frameIndex, {
          engine: selectedEngine,
          roi,
          seedPoints: selectedEngine === 'angiopy' ? seedPoints : undefined,
          returnProbability: true,
        });

        // Cache the result
        set((state) => {
          const newFrameData = new Map(state.frameData);
          const existing = newFrameData.get(frameIndex) || { ...EMPTY_FRAME_DATA };
          newFrameData.set(frameIndex, {
            ...existing,
            mask: result.mask,
            probabilityMap: result.probabilityMap,
          });
          return { isSegmenting: false, frameData: newFrameData };
        });

        return {
          frameIndex,
          mask: null,
          probabilityMap: null,
          centerline: [],
          width: result.width,
          height: result.height,
          engineUsed: result.engineUsed as SegmentationEngine,
          numSeedPoints: result.numSeedPoints,
        };

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Segmentation failed';
        set({ isSegmenting: false, error: message });
        return null;
      }
    },

    // Extract centerline
    extractCenterline: async (frameIndex, method = 'skeleton') => {
      try {
        const frameData = get().frameData.get(frameIndex);
        const result = await segmentationApi.extractCenterline(
          frameIndex,
          frameData?.mask || undefined,
          method
        );

        // Cache the result
        set((state) => {
          const newFrameData = new Map(state.frameData);
          const existing = newFrameData.get(frameIndex) || { ...EMPTY_FRAME_DATA };
          newFrameData.set(frameIndex, {
            ...existing,
            centerline: result.centerline,
            seedPoints: result.seedPoints,
          });
          return { frameData: newFrameData };
        });

        return result.centerline;

      } catch (error) {
        console.error('Centerline extraction failed:', error);
        return [];
      }
    },

    // Combined segment and extract - caches everything
    segmentAndExtract: async (frameIndex, roi) => {
      const { selectedEngine, seedPoints } = get();
      set({ isSegmenting: true, error: null });

      try {
        if (selectedEngine === 'angiopy') {
          if (seedPoints.length < 2) {
            throw new Error('AngioPy requires at least 2 seed points');
          }

          // Segment with AngioPy
          const segResult = await segmentationApi.segment(frameIndex, {
            engine: 'angiopy',
            seedPoints,
            returnProbability: true,
          });

          // Extract centerline
          const centerlineResult = await segmentationApi.extractCenterline(
            frameIndex,
            segResult.mask,
            'skeleton'
          );

          // Cache all results for this frame
          set((state) => {
            const newFrameData = new Map(state.frameData);
            newFrameData.set(frameIndex, {
              mask: segResult.mask,
              probabilityMap: segResult.probabilityMap,
              centerline: centerlineResult.centerline,
              seedPoints: centerlineResult.seedPoints,
            });
            return { isSegmenting: false, frameData: newFrameData };
          });

        } else {
          // nnU-Net: use combined endpoint
          const result = await segmentationApi.segmentAndExtract(frameIndex, roi);

          console.log('[segmentationStore] Caching frame', frameIndex, {
            mask: result.mask ? `${result.mask.length} chars` : 'null',
            centerline: result.centerline?.length ?? 0,
            seedPoints: result.seedPoints?.length ?? 0,
          });

          // Cache all results for this frame
          set((state) => {
            const newFrameData = new Map(state.frameData);
            newFrameData.set(frameIndex, {
              mask: result.mask,
              probabilityMap: result.probabilityMap,
              centerline: result.centerline,
              seedPoints: result.seedPoints,
            });
            return { isSegmenting: false, frameData: newFrameData };
          });
        }

      } catch (error) {
        const message = error instanceof Error ? error.message : 'Segmentation failed';
        set({ isSegmenting: false, error: message });
      }
    },

    // Getters - return cached data for specific frame
    getFrameData: (frameIndex) => get().frameData.get(frameIndex),

    getMask: (frameIndex) => {
      const data = get().frameData.get(frameIndex);
      return data?.mask ?? null;
    },

    getCenterline: (frameIndex) => {
      const data = get().frameData.get(frameIndex);
      return data?.centerline ?? [];
    },

    getSeedPoints: (frameIndex) => {
      const data = get().frameData.get(frameIndex);
      return data?.seedPoints ?? [];
    },

    hasData: (frameIndex) => {
      const data = get().frameData.get(frameIndex);
      return data !== undefined && (data.mask !== null || data.centerline.length > 0);
    },

    // Get all frames that have cached data
    getFramesWithData: () => {
      return Array.from(get().frameData.keys()).sort((a, b) => a - b);
    },

    // Clear single frame
    clearFrame: (frameIndex) => {
      set((state) => {
        const newFrameData = new Map(state.frameData);
        newFrameData.delete(frameIndex);
        return { frameData: newFrameData };
      });
    },

    // Clear all cached data
    clearAll: () => {
      set({
        frameData: new Map(),
        seedPoints: [],
      });
    },

    // Full reset
    reset: () => {
      set({
        selectedEngine: 'nnunet',
        isSegmenting: false,
        error: null,
        seedPoints: [],
        frameData: new Map(),
      });
    },
  }))
);

// Selectors for current frame (used with playerStore.currentFrame)
export const selectIsSegmenting = (state: SegmentationState) => state.isSegmenting;
export const selectSelectedEngine = (state: SegmentationState) => state.selectedEngine;
export const selectSeedPoints = (state: SegmentationState) => state.seedPoints;
export const selectAvailableEngines = (state: SegmentationState) => state.availableEngines;
export const selectFrameData = (state: SegmentationState) => state.frameData;
