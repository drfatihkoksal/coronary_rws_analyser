/**
 * QCA Panel Component
 *
 * Displays Quantitative Coronary Analysis metrics with method selection.
 */

import { useState, useCallback } from 'react';
import { useQcaStore, useCalibrationStore, usePlayerStore, useDicomStore } from '../../stores';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { api, QCAMethod } from '../../lib/api';

const QCA_METHODS: { value: QCAMethod; label: string; description: string }[] = [
  { value: 'gaussian', label: 'Gaussian', description: 'Best accuracy, sub-pixel fitting' },
  { value: 'parabolic', label: 'Parabolic', description: 'Fast, good accuracy' },
  { value: 'threshold', label: 'Threshold', description: 'Fastest, pixel-level' },
];

export function QCAPanel() {
  const currentMetrics = useQcaStore((s) => s.currentMetrics);
  const isCalculating = useQcaStore((s) => s.isCalculating);
  const setIsCalculating = useQcaStore((s) => s.setIsCalculating);
  const setCurrentMetrics = useQcaStore((s) => s.setCurrentMetrics);
  const isCalibrated = useCalibrationStore((s) => s.calibration !== null);
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const isLoaded = useDicomStore((s) => s.isLoaded);

  const [method, setMethod] = useState<QCAMethod>('gaussian');
  const [numPoints, setNumPoints] = useState(50);
  const [error, setError] = useState<string | null>(null);

  const handleCalculate = useCallback(async () => {
    if (!isLoaded) return;

    setError(null);
    setIsCalculating(true);

    try {
      const metrics = await api.qca.calculate(currentFrame, { numPoints, method });
      setCurrentMetrics(metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'QCA calculation failed');
    } finally {
      setIsCalculating(false);
    }
  }, [currentFrame, isLoaded, method, numPoints, setIsCalculating, setCurrentMetrics]);

  // Format number with units
  const formatMm = (value: number | undefined) => {
    if (value === undefined || value === null) return '-- mm';
    return `${value.toFixed(2)} mm`;
  };

  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null) return '-- %';
    return `${value.toFixed(1)}%`;
  };

  // Get stenosis severity
  const getStenosisSeverity = (ds: number): { label: string; variant: 'success' | 'warning' | 'danger' } => {
    if (ds < 50) return { label: 'Mild', variant: 'success' };
    if (ds < 70) return { label: 'Moderate', variant: 'warning' };
    return { label: 'Severe', variant: 'danger' };
  };

  return (
    <Card title="QCA Metrics" className="h-full overflow-auto">
      {/* Settings */}
      <div className="mb-3 space-y-2">
        <div className="flex gap-2">
          {QCA_METHODS.map((m) => (
            <button
              key={m.value}
              onClick={() => setMethod(m.value)}
              disabled={isCalculating}
              title={m.description}
              className={`flex-1 px-2 py-1 rounded text-xs transition-colors ${
                method === m.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">Points:</label>
          <select
            value={numPoints}
            onChange={(e) => setNumPoints(parseInt(e.target.value))}
            disabled={isCalculating}
            className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
          >
            <option value={30}>30</option>
            <option value={50}>50 (default)</option>
            <option value={70}>70</option>
          </select>
          <button
            onClick={handleCalculate}
            disabled={!isLoaded || isCalculating}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              isLoaded && !isCalculating
                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            {isCalculating ? 'Calc...' : 'Calculate'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-3 bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {isCalculating && (
        <div className="text-center text-gray-400 py-4">
          Calculating with {method} method...
        </div>
      )}

      {!isCalculating && !currentMetrics && (
        <div className="text-center text-gray-500 py-4 text-sm">
          Run segmentation then calculate QCA
        </div>
      )}

      {!isCalculating && currentMetrics && (
        <div className="space-y-3">
          {/* Calibration warning */}
          {!isCalibrated && (
            <div className="text-xs text-amber-400 bg-amber-900/30 p-2 rounded">
              ⚠️ Not calibrated - values may be inaccurate
            </div>
          )}

          {/* Key metrics */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <div className="text-gray-400 text-xs">MLD</div>
              <div className="font-semibold text-red-400">
                {formatMm(currentMetrics.mld)}
              </div>
            </div>
            <div>
              <div className="text-gray-400 text-xs">DS%</div>
              <div className="font-semibold">
                {formatPercent(currentMetrics.diameterStenosis)}
                {currentMetrics.diameterStenosis !== undefined && (
                  <Badge
                    variant={getStenosisSeverity(currentMetrics.diameterStenosis).variant}
                    className="ml-2"
                  >
                    {getStenosisSeverity(currentMetrics.diameterStenosis).label}
                  </Badge>
                )}
              </div>
            </div>
          </div>

          {/* Reference diameters */}
          <div className="border-t border-gray-700 pt-2">
            <div className="text-xs text-gray-400 mb-1">Reference Diameters</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex justify-between">
                <span className="text-cyan-400">Proximal:</span>
                <span>{formatMm(currentMetrics.proximalRd)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-yellow-400">Distal:</span>
                <span>{formatMm(currentMetrics.distalRd)}</span>
              </div>
            </div>
          </div>

          {/* Interpolated reference */}
          <div className="flex justify-between text-sm border-t border-gray-700 pt-2">
            <span className="text-gray-400">Interpolated RD:</span>
            <span>{formatMm(currentMetrics.interpolatedRd)}</span>
          </div>

          {/* Lesion length */}
          {currentMetrics.lesionLength !== null && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Lesion Length:</span>
              <span>{formatMm(currentMetrics.lesionLength)}</span>
            </div>
          )}

          {/* N-point info */}
          <div className="text-xs text-gray-500 border-t border-gray-700 pt-2">
            {currentMetrics.numPoints}-point analysis
            {currentMetrics.pixelSpacing && (
              <span className="ml-2">
                | {currentMetrics.pixelSpacing.toFixed(4)} mm/px
              </span>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
