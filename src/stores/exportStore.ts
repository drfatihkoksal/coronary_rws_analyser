/**
 * Export Store
 *
 * Manages export state and history.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

type ExportFormat = 'csv' | 'json' | 'pdf';
type ExportType = 'qca' | 'rws' | 'all';

interface ExportRecord {
  id: string;
  timestamp: Date;
  filename: string;
  format: ExportFormat;
  type: ExportType;
  sizeBytes: number;
  path: string;
}

interface ExportState {
  // Current export status
  isExporting: boolean;
  exportProgress: number;
  currentFormat: ExportFormat;
  currentType: ExportType;
  error: string | null;

  // Export history
  history: ExportRecord[];
  lastExportPath: string | null;

  // Actions
  startExport: (format: ExportFormat, type: ExportType) => void;
  setProgress: (progress: number) => void;
  completeExport: (record: Omit<ExportRecord, 'id' | 'timestamp'>) => void;
  failExport: (error: string) => void;
  clearHistory: () => void;
  reset: () => void;
}

export const useExportStore = create<ExportState>()(
  subscribeWithSelector((set, get) => ({
    isExporting: false,
    exportProgress: 0,
    currentFormat: 'csv',
    currentType: 'all',
    error: null,
    history: [],
    lastExportPath: null,

    startExport: (format, type) => {
      set({
        isExporting: true,
        exportProgress: 0,
        currentFormat: format,
        currentType: type,
        error: null,
      });
    },

    setProgress: (progress) => {
      set({ exportProgress: Math.min(100, Math.max(0, progress)) });
    },

    completeExport: (record) => {
      const newRecord: ExportRecord = {
        ...record,
        id: `export_${Date.now()}`,
        timestamp: new Date(),
      };

      set((state) => ({
        isExporting: false,
        exportProgress: 100,
        lastExportPath: record.path,
        history: [newRecord, ...state.history].slice(0, 50), // Keep last 50
      }));
    },

    failExport: (error) => {
      set({
        isExporting: false,
        exportProgress: 0,
        error,
      });
    },

    clearHistory: () => {
      set({ history: [] });
    },

    reset: () => {
      set({
        isExporting: false,
        exportProgress: 0,
        currentFormat: 'csv',
        currentType: 'all',
        error: null,
        history: [],
        lastExportPath: null,
      });
    },
  }))
);

// Selectors
export const selectIsExporting = (state: ExportState) => state.isExporting;
export const selectExportProgress = (state: ExportState) => state.exportProgress;
export const selectExportError = (state: ExportState) => state.error;
export const selectExportHistory = (state: ExportState) => state.history;
