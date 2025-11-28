/**
 * Export Panel Component
 *
 * Export analysis results to CSV/JSON formats.
 */

import { useState, useCallback } from 'react';
import { useExportStore, useQcaStore, useRwsStore } from '../../stores';
import { api } from '../../lib/api';

type ExportFormat = 'csv' | 'json';
type ExportType = 'qca' | 'rws' | 'all';

export function ExportPanel() {
  const isExporting = useExportStore((s) => s.isExporting);
  const exportProgress = useExportStore((s) => s.exportProgress);
  const exportHistory = useExportStore((s) => s.history);
  const startExport = useExportStore((s) => s.startExport);
  const completeExport = useExportStore((s) => s.completeExport);
  const failExport = useExportStore((s) => s.failExport);

  const hasQcaData = useQcaStore((s) => s.frameMetrics.size > 0);
  const hasRwsData = useRwsStore((s) => s.results.size > 0);

  const [format, setFormat] = useState<ExportFormat>('csv');
  const [exportType, setExportType] = useState<ExportType>('all');
  const [error, setError] = useState<string | null>(null);
  const [lastExport, setLastExport] = useState<{ filename: string; path: string } | null>(null);

  const handleExport = useCallback(async () => {
    setError(null);
    startExport(format, exportType);

    try {
      let response;

      if (exportType === 'qca') {
        response = await api.export.qca({ format, includeDiameterProfiles: true });
      } else if (exportType === 'rws') {
        response = await api.export.rws({ format, includeTemporalData: true });
      } else {
        response = await api.export.all({
          format,
          includeQca: true,
          includeRws: true,
          includeMetadata: true,
        });
      }

      if (response.success && response.filename && response.path) {
        completeExport({
          filename: response.filename,
          format,
          type: exportType,
          sizeBytes: response.sizeBytes || 0,
          path: response.path,
        });
        setLastExport({ filename: response.filename, path: response.path });
      } else {
        failExport(response.errorMessage || 'Export failed');
        setError(response.errorMessage || 'Export failed');
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Export failed';
      failExport(message);
      setError(message);
    }
  }, [format, exportType, startExport, completeExport, failExport]);

  const handleDownload = useCallback(async (filename: string) => {
    try {
      const url = `http://127.0.0.1:8000/export/download/${filename}`;
      window.open(url, '_blank');
    } catch (e) {
      setError('Failed to download file');
    }
  }, []);

  const canExport = hasQcaData || hasRwsData;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
        Export Results
      </h3>

      {/* Data Availability */}
      <div className="bg-gray-800 rounded p-3 space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-400">QCA Data:</span>
          <span className={hasQcaData ? 'text-green-400' : 'text-gray-500'}>
            {hasQcaData ? 'Available' : 'Not available'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">RWS Data:</span>
          <span className={hasRwsData ? 'text-green-400' : 'text-gray-500'}>
            {hasRwsData ? 'Available' : 'Not available'}
          </span>
        </div>
      </div>

      {/* Export Options */}
      <div className="space-y-3">
        {/* Format Selection */}
        <div>
          <label className="text-xs text-gray-400 block mb-1">Format</label>
          <div className="grid grid-cols-2 gap-2">
            {(['csv', 'json'] as ExportFormat[]).map((f) => (
              <button
                key={f}
                onClick={() => setFormat(f)}
                className={`px-3 py-2 rounded text-sm transition-colors ${
                  format === f
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Type Selection */}
        <div>
          <label className="text-xs text-gray-400 block mb-1">Data to Export</label>
          <div className="grid grid-cols-3 gap-1">
            {([
              { value: 'all', label: 'All' },
              { value: 'qca', label: 'QCA' },
              { value: 'rws', label: 'RWS' },
            ] as { value: ExportType; label: string }[]).map((option) => (
              <button
                key={option.value}
                onClick={() => setExportType(option.value)}
                disabled={
                  (option.value === 'qca' && !hasQcaData) ||
                  (option.value === 'rws' && !hasRwsData)
                }
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  exportType === option.value
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Export Button */}
        <button
          onClick={handleExport}
          disabled={!canExport || isExporting}
          className={`w-full py-2 rounded text-sm font-medium transition-colors ${
            canExport && !isExporting
              ? 'bg-green-600 hover:bg-green-500 text-white'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          {isExporting ? `Exporting... ${exportProgress.toFixed(0)}%` : 'Export'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Last Export */}
      {lastExport && (
        <div className="bg-green-900/30 border border-green-700 rounded p-2 space-y-2">
          <div className="text-xs text-green-400">Export successful!</div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-300 truncate flex-1">
              {lastExport.filename}
            </span>
            <button
              onClick={() => handleDownload(lastExport.filename)}
              className="text-xs text-blue-400 hover:text-blue-300 ml-2"
            >
              Download
            </button>
          </div>
        </div>
      )}

      {/* Export History */}
      {exportHistory.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-gray-400">Recent Exports</h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {exportHistory.slice(0, 5).map((record) => (
              <div
                key={record.id}
                className="flex items-center justify-between bg-gray-800 rounded px-2 py-1 text-xs"
              >
                <span className="text-gray-300 truncate flex-1">{record.filename}</span>
                <button
                  onClick={() => handleDownload(record.filename)}
                  className="text-blue-400 hover:text-blue-300 ml-2"
                >
                  Download
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
