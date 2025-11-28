/**
 * Segmentation Panel Component
 *
 * Engine selection and segmentation controls.
 * Supports nnU-Net (ROI-based) and AngioPy (seed-guided).
 */

import { useEffect } from 'react';
import { useSegmentationStore, usePlayerStore, useDicomStore, useAnnotationStore } from '../../stores';
import { Card, Button, Badge } from '../common';
import type { SegmentationEngine } from '../../types';

export function SegmentationPanel() {
  // Store state
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const currentFrame = usePlayerStore((s) => s.currentFrame);

  // Get ROI from annotation store (not tracking store!)
  // Subscribe to frameAnnotations to trigger re-render when ROI changes
  const frameAnnotations = useAnnotationStore((s) => s.frameAnnotations);
  const currentRoi = frameAnnotations.get(currentFrame)?.roi ?? null;

  const selectedEngine = useSegmentationStore((s) => s.selectedEngine);
  const availableEngines = useSegmentationStore((s) => s.availableEngines);
  const enginesLoaded = useSegmentationStore((s) => s.enginesLoaded);
  const isSegmenting = useSegmentationStore((s) => s.isSegmenting);
  const error = useSegmentationStore((s) => s.error);
  const seedPoints = useSegmentationStore((s) => s.seedPoints);
  const hasData = useSegmentationStore((s) => s.hasData);
  const getFramesWithData = useSegmentationStore((s) => s.getFramesWithData);

  const setEngine = useSegmentationStore((s) => s.setEngine);
  const loadEngines = useSegmentationStore((s) => s.loadEngines);
  const segmentAndExtract = useSegmentationStore((s) => s.segmentAndExtract);
  const clearSeedPoints = useSegmentationStore((s) => s.clearSeedPoints);
  const removeSeedPoint = useSegmentationStore((s) => s.removeSeedPoint);

  // Load engines on mount
  useEffect(() => {
    if (!enginesLoaded) {
      loadEngines();
    }
  }, [enginesLoaded, loadEngines]);

  // Check if can segment
  const canSegment = isLoaded && !isSegmenting && (
    (selectedEngine === 'nnunet') ||
    (selectedEngine === 'angiopy' && seedPoints.length >= 2)
  );

  // Handle segment
  const handleSegment = async () => {
    if (!canSegment) return;

    if (selectedEngine === 'nnunet') {
      await segmentAndExtract(currentFrame, currentRoi ?? undefined);
    } else {
      await segmentAndExtract(currentFrame);
    }
  };

  // Handle engine change
  const handleEngineChange = (engine: SegmentationEngine) => {
    setEngine(engine);
    // Clear seed points when switching engines
    if (engine !== 'angiopy') {
      clearSeedPoints();
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-white">Segmentation</h3>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-auto">
        {/* Engine Selection */}
        <div>
          <label className="text-xs text-gray-400 block mb-1">Engine</label>
          <div className="flex gap-1">
            <button
              onClick={() => handleEngineChange('nnunet')}
              disabled={!isLoaded || availableEngines?.nnunet?.available === false}
              className={`
                flex-1 px-2 py-1.5 text-xs rounded transition-colors
                ${selectedEngine === 'nnunet'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }
                ${(!isLoaded || availableEngines?.nnunet?.available === false) ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              nnU-Net
            </button>
            <button
              onClick={() => handleEngineChange('angiopy')}
              disabled={!isLoaded || availableEngines?.angiopy?.available === false}
              className={`
                flex-1 px-2 py-1.5 text-xs rounded transition-colors
                ${selectedEngine === 'angiopy'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }
                ${(!isLoaded || availableEngines?.angiopy?.available === false) ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              AngioPy
            </button>
          </div>
        </div>

        {/* Engine Info */}
        {availableEngines && (
          <div className="text-xs text-gray-400 bg-gray-800 p-2 rounded">
            {selectedEngine === 'nnunet' ? (
              <>
                <p className="font-medium text-gray-300">ROI-based segmentation</p>
                <p className="mt-1">Draw ROI (B key) then segment</p>
                {currentRoi && (
                  <p className="text-green-400 mt-1">ROI ready</p>
                )}
              </>
            ) : (
              <>
                <p className="font-medium text-gray-300">Seed point-guided</p>
                <p className="mt-1">Place 2-10 points (S key) along vessel</p>
                <p className="mt-1">Order: proximal to distal</p>
              </>
            )}
          </div>
        )}

        {/* Seed Points (AngioPy only) */}
        {selectedEngine === 'angiopy' && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-400">
                Seed Points ({seedPoints.length}/10)
              </label>
              {seedPoints.length > 0 && (
                <button
                  onClick={clearSeedPoints}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Clear all
                </button>
              )}
            </div>

            {seedPoints.length === 0 ? (
              <div className="text-xs text-gray-500 bg-gray-800 p-2 rounded">
                Press <kbd className="bg-gray-700 px-1 rounded">S</kbd> and click on vessel to add points
              </div>
            ) : (
              <div className="space-y-1 max-h-32 overflow-auto">
                {seedPoints.map((point, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-800 px-2 py-1 rounded text-xs"
                  >
                    <span className="text-gray-300">
                      #{index + 1}: ({point.x.toFixed(0)}, {point.y.toFixed(0)})
                    </span>
                    <button
                      onClick={() => removeSeedPoint(index)}
                      className="text-gray-500 hover:text-red-400"
                    >
                      x
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Validation message */}
            {seedPoints.length === 1 && (
              <p className="text-xs text-yellow-400 mt-1">
                Need at least 2 points
              </p>
            )}
            {seedPoints.length >= 2 && (
              <p className="text-xs text-green-400 mt-1">
                Ready to segment
              </p>
            )}
          </div>
        )}

        {/* nnU-Net ROI status */}
        {selectedEngine === 'nnunet' && (
          <div>
            <label className="text-xs text-gray-400 block mb-1">ROI Status</label>
            {currentRoi ? (
              <div className="text-xs bg-gray-800 p-2 rounded">
                <span className="text-green-400">ROI defined</span>
                <span className="text-gray-500 ml-2">
                  {Math.round(currentRoi.width)}x{Math.round(currentRoi.height)} at ({Math.round(currentRoi.x)}, {Math.round(currentRoi.y)})
                </span>
              </div>
            ) : (
              <div className="text-xs text-gray-500 bg-gray-800 p-2 rounded">
                Press <kbd className="bg-gray-700 px-1 rounded">B</kbd> and draw ROI
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <Badge variant="danger" className="w-full justify-center">
            {error}
          </Badge>
        )}

        {/* Segment Button */}
        <Button
          onClick={handleSegment}
          disabled={!canSegment}
          variant="primary"
          className="w-full"
        >
          {isSegmenting ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin">⏳</span>
              Segmenting...
            </span>
          ) : (
            `Segment (${selectedEngine})`
          )}
        </Button>

        {/* Result status */}
        {hasData(currentFrame) && (
          <div className="text-xs text-green-400 text-center">
            Frame {currentFrame} segmented
          </div>
        )}

        {/* Cached frames info */}
        {getFramesWithData().length > 0 && (
          <div className="text-xs text-gray-400 text-center">
            {getFramesWithData().length} frame(s) cached
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="p-2 border-t border-gray-700 text-xs text-gray-500">
        Frame: {currentFrame} | Engine: {selectedEngine}
        {availableEngines && (
          <span className="ml-2">
            {availableEngines.nnunet?.available && availableEngines.angiopy?.available
              ? '(Both available)'
              : availableEngines.nnunet?.available
              ? '(nnU-Net only)'
              : availableEngines.angiopy?.available
              ? '(AngioPy only)'
              : '(None available)'}
          </span>
        )}
      </div>
    </Card>
  );
}
