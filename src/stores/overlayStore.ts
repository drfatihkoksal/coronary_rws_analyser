/**
 * Overlay Visibility Store
 *
 * Controls visibility of different overlay layers on the canvas.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface OverlayVisibility {
  mask: boolean;
  centerline: boolean;
  seedPoints: boolean;
  yoloKeypoints: boolean;
  roi: boolean;
  diameterMarkers: boolean;
}

interface OverlayState {
  // Visibility toggles
  visibility: OverlayVisibility;

  // Actions
  setVisibility: (key: keyof OverlayVisibility, visible: boolean) => void;
  toggleVisibility: (key: keyof OverlayVisibility) => void;
  setAllVisible: (visible: boolean) => void;
  reset: () => void;
}

const DEFAULT_VISIBILITY: OverlayVisibility = {
  mask: true,
  centerline: true,
  seedPoints: true,
  yoloKeypoints: true,
  roi: true,
  diameterMarkers: true,
};

export const useOverlayStore = create<OverlayState>()(
  subscribeWithSelector((set) => ({
    visibility: { ...DEFAULT_VISIBILITY },

    setVisibility: (key, visible) => {
      set((state) => ({
        visibility: { ...state.visibility, [key]: visible },
      }));
    },

    toggleVisibility: (key) => {
      set((state) => ({
        visibility: { ...state.visibility, [key]: !state.visibility[key] },
      }));
    },

    setAllVisible: (visible) => {
      set({
        visibility: {
          mask: visible,
          centerline: visible,
          seedPoints: visible,
          yoloKeypoints: visible,
          roi: visible,
          diameterMarkers: visible,
        },
      });
    },

    reset: () => {
      set({ visibility: { ...DEFAULT_VISIBILITY } });
    },
  }))
);

// Selectors
export const selectOverlayVisibility = (state: OverlayState) => state.visibility;
export const selectMaskVisible = (state: OverlayState) => state.visibility.mask;
export const selectCenterlineVisible = (state: OverlayState) => state.visibility.centerline;
export const selectSeedPointsVisible = (state: OverlayState) => state.visibility.seedPoints;
export const selectYoloKeypointsVisible = (state: OverlayState) => state.visibility.yoloKeypoints;
export const selectRoiVisible = (state: OverlayState) => state.visibility.roi;
