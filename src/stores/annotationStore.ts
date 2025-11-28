/**
 * Annotation Store
 *
 * Manages seed points and ROI annotations per frame.
 * Separate from segmentationStore for cleaner state management.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Point, BoundingBox } from '../types';

interface FrameAnnotation {
  seedPoints: Point[];
  roi: BoundingBox | null;
}

interface AnnotationState {
  // Per-frame annotations (Map for sparse storage)
  frameAnnotations: Map<number, FrameAnnotation>;

  // Current mode
  seedMode: boolean;
  roiMode: boolean;

  // Active frame being annotated
  activeFrame: number | null;

  // Actions - Seed Points
  getSeedPoints: (frameIndex: number) => Point[];
  addSeedPoint: (frameIndex: number, point: Point) => void;
  updateSeedPoint: (frameIndex: number, index: number, point: Point) => void;
  removeSeedPoint: (frameIndex: number, index: number) => void;
  clearSeedPoints: (frameIndex: number) => void;
  setSeedMode: (enabled: boolean) => void;

  // Actions - ROI
  getROI: (frameIndex: number) => BoundingBox | null;
  setROI: (frameIndex: number, roi: BoundingBox) => void;
  clearROI: (frameIndex: number) => void;
  setROIMode: (enabled: boolean) => void;

  // Copy annotations to another frame
  copyAnnotations: (fromFrame: number, toFrame: number) => void;

  // Clear all
  clearAllAnnotations: () => void;

  // Reset
  reset: () => void;
}

const getOrCreateAnnotation = (
  map: Map<number, FrameAnnotation>,
  frameIndex: number
): FrameAnnotation => {
  let annotation = map.get(frameIndex);
  if (!annotation) {
    annotation = { seedPoints: [], roi: null };
    map.set(frameIndex, annotation);
  }
  return annotation;
};

export const useAnnotationStore = create<AnnotationState>()(
  subscribeWithSelector((set, get) => ({
    frameAnnotations: new Map(),
    seedMode: false,
    roiMode: false,
    activeFrame: null,

    // Seed Points
    getSeedPoints: (frameIndex) => {
      const annotation = get().frameAnnotations.get(frameIndex);
      return annotation?.seedPoints || [];
    },

    addSeedPoint: (frameIndex, point) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = getOrCreateAnnotation(newMap, frameIndex);
        annotation.seedPoints = [...annotation.seedPoints, point];
        return { frameAnnotations: newMap };
      });
    },

    updateSeedPoint: (frameIndex, index, point) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = newMap.get(frameIndex);
        if (annotation && index >= 0 && index < annotation.seedPoints.length) {
          const newPoints = [...annotation.seedPoints];
          newPoints[index] = point;
          annotation.seedPoints = newPoints;
        }
        return { frameAnnotations: newMap };
      });
    },

    removeSeedPoint: (frameIndex, index) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = newMap.get(frameIndex);
        if (annotation && index >= 0 && index < annotation.seedPoints.length) {
          annotation.seedPoints = annotation.seedPoints.filter((_, i) => i !== index);
        }
        return { frameAnnotations: newMap };
      });
    },

    clearSeedPoints: (frameIndex) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = newMap.get(frameIndex);
        if (annotation) {
          annotation.seedPoints = [];
        }
        return { frameAnnotations: newMap };
      });
    },

    setSeedMode: (enabled) => {
      set({ seedMode: enabled, roiMode: enabled ? false : get().roiMode });
    },

    // ROI
    getROI: (frameIndex) => {
      const annotation = get().frameAnnotations.get(frameIndex);
      return annotation?.roi || null;
    },

    setROI: (frameIndex, roi) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = getOrCreateAnnotation(newMap, frameIndex);
        annotation.roi = roi;
        return { frameAnnotations: newMap };
      });
    },

    clearROI: (frameIndex) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const annotation = newMap.get(frameIndex);
        if (annotation) {
          annotation.roi = null;
        }
        return { frameAnnotations: newMap };
      });
    },

    setROIMode: (enabled) => {
      set({ roiMode: enabled, seedMode: enabled ? false : get().seedMode });
    },

    // Copy annotations
    copyAnnotations: (fromFrame, toFrame) => {
      set((state) => {
        const newMap = new Map(state.frameAnnotations);
        const source = newMap.get(fromFrame);
        if (source) {
          newMap.set(toFrame, {
            seedPoints: [...source.seedPoints],
            roi: source.roi ? { ...source.roi } : null,
          });
        }
        return { frameAnnotations: newMap };
      });
    },

    // Clear all
    clearAllAnnotations: () => {
      set({ frameAnnotations: new Map() });
    },

    // Reset
    reset: () => {
      set({
        frameAnnotations: new Map(),
        seedMode: false,
        roiMode: false,
        activeFrame: null,
      });
    },
  }))
);

// Selectors
export const selectSeedMode = (state: AnnotationState) => state.seedMode;
export const selectROIMode = (state: AnnotationState) => state.roiMode;
export const selectFrameAnnotations = (state: AnnotationState) => state.frameAnnotations;
