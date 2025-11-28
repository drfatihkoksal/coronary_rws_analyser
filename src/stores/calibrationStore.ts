/**
 * Calibration Store
 *
 * Manages pixel-to-mm calibration for QCA measurements.
 * Sources: DICOM metadata or manual catheter calibration.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Calibration, CalibrationSource } from '../types';
import { calibrationApi } from '../lib/api';
import { useDicomStore } from './dicomStore';

// Catheter sizes in French (Fr) with corresponding diameters in mm
export const CATHETER_SIZES: Record<number, number> = {
  5: 1.67,  // 5Fr = 1.67mm
  6: 2.0,   // 6Fr = 2.0mm
  7: 2.33,  // 7Fr = 2.33mm
  8: 2.67,  // 8Fr = 2.67mm
};

interface CalibrationState {
  // Current calibration
  calibration: Calibration | null;

  // Calibration source
  source: CalibrationSource | null;

  // Manual calibration state
  isCalibrating: boolean;
  manualCatheterSize: number | null;
  manualMeasuredPixels: number | null;

  // Actions
  setFromDicom: () => void;
  setManual: (catheterSizeFr: number, measuredPixels: number) => Promise<void>;
  setCustom: (pixelSpacing: [number, number]) => void;

  // Getters
  getPixelSpacing: () => number;
  convertPixelsToMm: (pixels: number) => number;
  convertMmToPixels: (mm: number) => number;

  // Reset
  reset: () => void;
}

export const useCalibrationStore = create<CalibrationState>()(
  persist(
    (set, get) => ({
      // Initial state
      calibration: null,
      source: null,
      isCalibrating: false,
      manualCatheterSize: null,
      manualMeasuredPixels: null,

      // Set calibration from DICOM metadata
      setFromDicom: () => {
        const dicomMetadata = useDicomStore.getState().metadata;

        if (dicomMetadata?.pixelSpacing) {
          set({
            calibration: {
              pixelSpacing: dicomMetadata.pixelSpacing as [number, number],
              source: 'dicom',
              catheterSizeFr: null,
            },
            source: 'dicom',
          });
        }
      },

      // Set manual catheter-based calibration
      setManual: async (catheterSizeFr, measuredPixels) => {
        set({ isCalibrating: true });

        try {
          // Calculate pixel spacing from catheter
          const knownDiameter = CATHETER_SIZES[catheterSizeFr];
          if (!knownDiameter) {
            throw new Error(`Unknown catheter size: ${catheterSizeFr}Fr`);
          }

          const pixelSpacing = knownDiameter / measuredPixels;

          // Send to backend
          await calibrationApi.setManual(catheterSizeFr, measuredPixels);

          set({
            isCalibrating: false,
            calibration: {
              pixelSpacing: [pixelSpacing, pixelSpacing],
              source: 'manual_catheter',
              catheterSizeFr,
            },
            source: 'manual_catheter',
            manualCatheterSize: catheterSizeFr,
            manualMeasuredPixels: measuredPixels,
          });

        } catch (error) {
          console.error('Manual calibration failed:', error);
          set({ isCalibrating: false });
          throw error;
        }
      },

      // Set custom pixel spacing
      setCustom: (pixelSpacing) => {
        set({
          calibration: {
            pixelSpacing,
            source: 'manual',
            catheterSizeFr: null,
          },
          source: 'manual',
        });
      },

      // Get current pixel spacing (mm/pixel)
      getPixelSpacing: () => {
        const { calibration } = get();
        if (calibration?.pixelSpacing) {
          return calibration.pixelSpacing[0]; // Use row spacing
        }
        return 1.0; // Default: 1 pixel = 1 unit (uncalibrated)
      },

      // Convert pixels to mm
      convertPixelsToMm: (pixels) => {
        const pixelSpacing = get().getPixelSpacing();
        return pixels * pixelSpacing;
      },

      // Convert mm to pixels
      convertMmToPixels: (mm) => {
        const pixelSpacing = get().getPixelSpacing();
        return mm / pixelSpacing;
      },

      // Reset
      reset: () => {
        set({
          calibration: null,
          source: null,
          isCalibrating: false,
          manualCatheterSize: null,
          manualMeasuredPixels: null,
        });
      },
    }),
    {
      name: 'calibration-storage',
      partialize: (state) => ({
        calibration: state.calibration,
        source: state.source,
        manualCatheterSize: state.manualCatheterSize,
        manualMeasuredPixels: state.manualMeasuredPixels,
      }),
    }
  )
);

// Selectors
export const selectCalibration = (state: CalibrationState) => state.calibration;
export const selectSource = (state: CalibrationState) => state.source;
export const selectIsCalibrated = (state: CalibrationState) => state.calibration !== null;
