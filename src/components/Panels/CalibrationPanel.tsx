/**
 * Calibration Panel Component
 *
 * Catheter-based pixel spacing calibration.
 * Uses seed points placed on catheter for measurement.
 */

import { useState, useCallback } from 'react';
import { useCalibrationStore, useSegmentationStore, usePlayerStore, useDicomStore } from '../../stores';
import { api } from '../../lib/api';

const CATHETER_SIZES = {
  '5F': 1.67,
  '6F': 2.00,
  '7F': 2.33,
  '8F': 2.67,
};

export function CalibrationPanel() {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const seedPoints = useSegmentationStore((s) => s.seedPoints);

  const pixelSpacing = useCalibrationStore((s) => s.pixelSpacing);
  const source = useCalibrationStore((s) => s.source);
  const isCalibrated = useCalibrationStore((s) => s.isCalibrated);
  const setCalibration = useCalibrationStore((s) => s.setCalibration);

  const [selectedSize, setSelectedSize] = useState<string>('6F');
  const [customSize, setCustomSize] = useState<string>('2.0');
  const [useCustom, setUseCustom] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    newSpacing: number;
    oldSpacing: number | null;
    quality: number;
  } | null>(null);

  const handleCalibrate = useCallback(async () => {
    if (seedPoints.length < 2) {
      setError('Place at least 2 seed points on the catheter');
      return;
    }

    setIsCalibrating(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.calibration.manual({
        catheterSize: useCustom ? 'custom' : selectedSize,
        customSizeMm: useCustom ? parseFloat(customSize) : undefined,
        seedPoints: seedPoints.map((p) => [p.x, p.y]),
        frameIndex: currentFrame,
        method: 'gaussian',
        nPoints: 50,
      });

      if (response.success && response.newPixelSpacing) {
        setCalibration(response.newPixelSpacing, 'manual');
        setResult({
          newSpacing: response.newPixelSpacing,
          oldSpacing: response.oldPixelSpacing || null,
          quality: response.qualityScore || 1.0,
        });
      } else {
        setError(response.errorMessage || 'Calibration failed');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Calibration failed');
    } finally {
      setIsCalibrating(false);
    }
  }, [seedPoints, currentFrame, selectedSize, customSize, useCustom, setCalibration]);

  const handleFromDicom = useCallback(async () => {
    setIsCalibrating(true);
    setError(null);

    try {
      const response = await api.calibration.fromDicom();
      if (response.pixelSpacing) {
        setCalibration(response.pixelSpacing, 'dicom');
        setResult({
          newSpacing: response.pixelSpacing,
          oldSpacing: null,
          quality: response.confidence === 'high' ? 1.0 : 0.7,
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to extract from DICOM');
    } finally {
      setIsCalibrating(false);
    }
  }, [setCalibration]);

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
        Calibration
      </h3>

      {/* Current Calibration Status */}
      <div className="bg-gray-800 rounded p-3 space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Pixel Spacing:</span>
          <span className={isCalibrated ? 'text-green-400' : 'text-yellow-400'}>
            {pixelSpacing.toFixed(4)} mm/px
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Source:</span>
          <span className="text-gray-300 capitalize">{source}</span>
        </div>
      </div>

      {/* DICOM Calibration */}
      <button
        onClick={handleFromDicom}
        disabled={!isLoaded || isCalibrating}
        className="w-full py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded text-sm transition-colors"
      >
        Extract from DICOM
      </button>

      {/* Manual Calibration */}
      <div className="space-y-3">
        <h4 className="text-xs font-medium text-gray-400">Manual Catheter Calibration</h4>

        {/* Catheter Size Selection */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="useCustom"
              checked={useCustom}
              onChange={(e) => setUseCustom(e.target.checked)}
              className="rounded bg-gray-700 border-gray-600"
            />
            <label htmlFor="useCustom" className="text-xs text-gray-400">
              Custom size
            </label>
          </div>

          {useCustom ? (
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={customSize}
                onChange={(e) => setCustomSize(e.target.value)}
                step="0.01"
                min="0.5"
                max="5"
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white"
              />
              <span className="text-gray-400 text-sm">mm</span>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-1">
              {Object.entries(CATHETER_SIZES).map(([size, diameter]) => (
                <button
                  key={size}
                  onClick={() => setSelectedSize(size)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    selectedSize === size
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  title={`${diameter} mm`}
                >
                  {size}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="text-xs text-gray-500 space-y-1">
          <p>1. Press <kbd className="bg-gray-700 px-1 rounded">S</kbd> to enter seed mode</p>
          <p>2. Place 2+ points along the catheter</p>
          <p>3. Click Calibrate</p>
        </div>

        {/* Seed Point Status */}
        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Seed Points:</span>
          <span className={seedPoints.length >= 2 ? 'text-green-400' : 'text-yellow-400'}>
            {seedPoints.length} {seedPoints.length >= 2 ? '✓' : '(need 2+)'}
          </span>
        </div>

        {/* Calibrate Button */}
        <button
          onClick={handleCalibrate}
          disabled={!isLoaded || isCalibrating || seedPoints.length < 2}
          className={`w-full py-2 rounded text-sm font-medium transition-colors ${
            isLoaded && seedPoints.length >= 2 && !isCalibrating
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          {isCalibrating ? 'Calibrating...' : 'Calibrate'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="bg-green-900/30 border border-green-700 rounded p-2 space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-green-400">New Spacing:</span>
            <span className="text-white">{result.newSpacing.toFixed(4)} mm/px</span>
          </div>
          {result.oldSpacing && (
            <div className="flex justify-between">
              <span className="text-green-400">Previous:</span>
              <span className="text-gray-400">{result.oldSpacing.toFixed(4)} mm/px</span>
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-green-400">Quality:</span>
            <span className={result.quality > 0.7 ? 'text-green-400' : 'text-yellow-400'}>
              {(result.quality * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
