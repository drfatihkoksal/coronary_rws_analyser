/**
 * RWS Panel Component (Primary Feature)
 *
 * Displays Radial Wall Strain analysis results with outlier detection options.
 * RWS = (Dmax - Dmin) / Dmax × 100%
 */

import { useState, useCallback } from 'react';
import { useRwsStore, useEcgStore, getRwsColor, getRwsLabel } from '../../stores';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import { api, OutlierMethod } from '../../lib/api';
import type { RWSInterpretation } from '../../types';

const OUTLIER_METHODS: { value: OutlierMethod; label: string; description: string }[] = [
  { value: 'none', label: 'None', description: 'No outlier filtering' },
  { value: 'hampel', label: 'Hampel', description: 'Robust MAD-based filter (recommended)' },
  { value: 'double_hampel', label: 'Double', description: 'Two-pass Hampel filter' },
  { value: 'iqr', label: 'IQR', description: 'Interquartile range' },
  { value: 'temporal', label: 'Temporal', description: 'Frame-to-frame consistency' },
];

export function RWSPanel() {
  const currentResult = useRwsStore((s) => s.currentResult);
  const results = useRwsStore((s) => s.results);
  const summary = useRwsStore((s) => s.summary);
  const isCalculating = useRwsStore((s) => s.isCalculating);
  const setIsCalculating = useRwsStore((s) => s.setIsCalculating);
  const setCurrentResult = useRwsStore((s) => s.setCurrentResult);
  const clearResults = useRwsStore((s) => s.clearResults);

  const selectedBeat = useEcgStore((s) => s.selectedBeat);

  const [outlierMethod, setOutlierMethod] = useState<OutlierMethod>('hampel');
  const [error, setError] = useState<string | null>(null);

  const handleCalculate = useCallback(async () => {
    if (!selectedBeat) {
      setError('Select a cardiac beat first (use ECG panel or Frame Range)');
      return;
    }

    setError(null);
    setIsCalculating(true);

    try {
      const result = await api.rws.calculate(
        selectedBeat.startFrame,
        selectedBeat.endFrame,
        { beatNumber: selectedBeat.beatNumber, outlierMethod }
      );
      setCurrentResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'RWS calculation failed');
    } finally {
      setIsCalculating(false);
    }
  }, [selectedBeat, outlierMethod, setIsCalculating, setCurrentResult]);

  // Format RWS value
  const formatRws = (value: number | undefined) => {
    if (value === undefined || value === null) return '--';
    return value.toFixed(1);
  };

  // Get badge variant from interpretation
  const getVariant = (interp: RWSInterpretation): 'success' | 'warning' | 'danger' => {
    switch (interp) {
      case 'normal':
        return 'success';
      case 'intermediate':
        return 'warning';
      case 'elevated':
        return 'danger';
    }
  };

  return (
    <Card title="RWS Analysis" className="h-full overflow-auto">
      {/* Header badge */}
      <div className="flex items-center justify-between mb-3">
        <Badge variant="info">Primary Feature</Badge>
        {results.length > 0 && (
          <Button variant="ghost" size="sm" onClick={clearResults}>
            Clear
          </Button>
        )}
      </div>

      {/* Outlier Method Selection */}
      <div className="mb-3 space-y-2">
        <div className="text-xs text-gray-400 mb-1">Outlier Detection:</div>
        <div className="grid grid-cols-3 gap-1">
          {OUTLIER_METHODS.slice(0, 3).map((m) => (
            <button
              key={m.value}
              onClick={() => setOutlierMethod(m.value)}
              disabled={isCalculating}
              title={m.description}
              className={`px-2 py-1 rounded text-xs transition-colors ${
                outlierMethod === m.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-1">
          {OUTLIER_METHODS.slice(3).map((m) => (
            <button
              key={m.value}
              onClick={() => setOutlierMethod(m.value)}
              disabled={isCalculating}
              title={m.description}
              className={`px-2 py-1 rounded text-xs transition-colors ${
                outlierMethod === m.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Calculate Button */}
        <button
          onClick={handleCalculate}
          disabled={isCalculating || !selectedBeat}
          className={`w-full py-2 rounded text-sm font-medium transition-colors ${
            !isCalculating && selectedBeat
              ? 'bg-green-600 hover:bg-green-500 text-white'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          {isCalculating ? 'Calculating...' : 'Calculate RWS'}
        </button>

        {!selectedBeat && (
          <div className="text-xs text-amber-400">
            Select frame range first
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-3 bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {isCalculating && (
        <div className="text-center text-gray-400 py-4">
          Calculating RWS with {outlierMethod} filter...
        </div>
      )}

      {!isCalculating && !currentResult && !error && (
        <div className="text-center text-gray-500 py-4 text-sm">
          <p>Calculate QCA for multiple frames</p>
          <p>then run RWS analysis</p>
        </div>
      )}

      {!isCalculating && currentResult && (
        <div className="space-y-4">
          {/* MLD RWS - Most important */}
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">MLD RWS</span>
              <Badge variant={getVariant(currentResult.mldRws.interpretation)}>
                {getRwsLabel(currentResult.mldRws.interpretation)}
              </Badge>
            </div>
            <div
              className="text-3xl font-bold"
              style={{ color: getRwsColor(currentResult.mldRws.interpretation) }}
            >
              {formatRws(currentResult.mldRws.rws)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Dmax: {currentResult.mldRws.dmax.toFixed(2)}mm (F{currentResult.mldRws.dmaxFrame})
              {' | '}
              Dmin: {currentResult.mldRws.dmin.toFixed(2)}mm (F{currentResult.mldRws.dminFrame})
            </div>
          </div>

          {/* Reference RWS values */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-gray-900 rounded p-2">
              <div className="text-gray-400 text-xs">Proximal RWS</div>
              <div
                className="font-semibold"
                style={{ color: getRwsColor(currentResult.proximalRws.interpretation) }}
              >
                {formatRws(currentResult.proximalRws.rws)}%
              </div>
            </div>
            <div className="bg-gray-900 rounded p-2">
              <div className="text-gray-400 text-xs">Distal RWS</div>
              <div
                className="font-semibold"
                style={{ color: getRwsColor(currentResult.distalRws.interpretation) }}
              >
                {formatRws(currentResult.distalRws.rws)}%
              </div>
            </div>
          </div>

          {/* Average */}
          <div className="flex justify-between text-sm border-t border-gray-700 pt-2">
            <span className="text-gray-400">Average RWS:</span>
            <span>{formatRws(currentResult.averageRws)}%</span>
          </div>

          {/* Frame range */}
          <div className="text-xs text-gray-500">
            Frames {currentResult.startFrame} - {currentResult.endFrame}
            ({currentResult.numFrames} frames)
            {currentResult.beatNumber !== null && (
              <span> | Beat #{currentResult.beatNumber}</span>
            )}
          </div>
        </div>
      )}

      {/* Summary (if multiple beats) */}
      {summary && results.length > 1 && (
        <div className="mt-4 border-t border-gray-700 pt-3">
          <div className="text-xs text-gray-400 mb-2">
            Summary ({summary.numBeats} beats)
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Mean MLD RWS:</span>
              <span className="ml-1">{summary.mldRwsMean.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Std:</span>
              <span className="ml-1">±{summary.mldRwsStd.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Min:</span>
              <span className="ml-1">{summary.mldRwsMin.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Max:</span>
              <span className="ml-1">{summary.mldRwsMax.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Clinical reference */}
      <div className="mt-4 p-2 bg-gray-900 rounded text-xs text-gray-500">
        <div className="font-medium text-gray-400 mb-1">Clinical Thresholds</div>
        <div className="flex justify-between">
          <span className="text-green-400">Normal: &lt;8%</span>
          <span className="text-amber-400">Intermediate: 8-12%</span>
          <span className="text-red-400">Elevated: &gt;12%</span>
        </div>
        <div className="mt-1 text-gray-600">Ref: Hong et al., EuroIntervention 2023</div>
      </div>
    </Card>
  );
}
