/**
 * Frame Range Selection Component
 *
 * Allows selection of frame ranges for RWS calculation.
 * Supports manual selection and ECG beat-based selection.
 * Updates ecgStore.selectedBeat for RWSPanel integration.
 */

import { useState, useCallback, useEffect } from 'react';
import { usePlayerStore, useEcgStore, useRwsStore } from '../../stores';

interface FrameRangeSelectionProps {
  onCalculateRws?: (startFrame: number, endFrame: number) => void;
}

export function FrameRangeSelection({ onCalculateRws }: FrameRangeSelectionProps) {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);

  const beatBoundaries = useEcgStore((s) => s.beatBoundaries);
  const rPeaks = useEcgStore((s) => s.ecgData?.rPeaks);
  const setSelectedBeatInStore = useEcgStore((s) => s.setSelectedBeat);

  const isCalculating = useRwsStore((s) => s.isCalculating);

  const [startFrame, setStartFrame] = useState<number>(0);
  const [endFrame, setEndFrame] = useState<number>(totalFrames > 0 ? totalFrames - 1 : 0);
  const [selectedBeatIndex, setSelectedBeatIndex] = useState<number | null>(null);

  // Sync with ecgStore whenever frame range changes
  useEffect(() => {
    if (startFrame < endFrame) {
      setSelectedBeatInStore({
        startFrame,
        endFrame,
        beatNumber: selectedBeatIndex !== null ? selectedBeatIndex + 1 : null,
      });
    }
  }, [startFrame, endFrame, selectedBeatIndex, setSelectedBeatInStore]);

  // Set current frame as start
  const setCurrentAsStart = useCallback(() => {
    setStartFrame(currentFrame);
    setSelectedBeatIndex(null);
  }, [currentFrame]);

  // Set current frame as end
  const setCurrentAsEnd = useCallback(() => {
    setEndFrame(currentFrame);
    setSelectedBeatIndex(null);
  }, [currentFrame]);

  // Select beat from ECG
  const selectBeat = useCallback((beatIndex: number) => {
    if (beatBoundaries && beatIndex >= 0 && beatIndex < beatBoundaries.length) {
      const beat = beatBoundaries[beatIndex];
      setStartFrame(beat.startFrame);
      setEndFrame(beat.endFrame);
      setSelectedBeatIndex(beatIndex);
    }
  }, [beatBoundaries]);

  // Calculate RWS
  const handleCalculate = useCallback(() => {
    if (onCalculateRws && startFrame < endFrame) {
      onCalculateRws(startFrame, endFrame);
    }
  }, [onCalculateRws, startFrame, endFrame]);

  const frameCount = endFrame - startFrame + 1;
  const hasValidRange = startFrame < endFrame && frameCount >= 2;
  const hasBeatData = beatBoundaries && beatBoundaries.length > 0;

  return (
    <div className="space-y-4">
      {/* Manual Frame Selection */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-gray-300">Frame Range</h4>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Start Frame</label>
            <div className="flex gap-1">
              <input
                type="number"
                value={startFrame}
                onChange={(e) => {
                  setStartFrame(parseInt(e.target.value) || 0);
                  setSelectedBeatIndex(null);
                }}
                min={0}
                max={totalFrames - 1}
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white"
              />
              <button
                onClick={setCurrentAsStart}
                className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs text-gray-300"
                title="Use current frame"
              >
                Current
              </button>
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">End Frame</label>
            <div className="flex gap-1">
              <input
                type="number"
                value={endFrame}
                onChange={(e) => {
                  setEndFrame(parseInt(e.target.value) || 0);
                  setSelectedBeatIndex(null);
                }}
                min={0}
                max={totalFrames - 1}
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white"
              />
              <button
                onClick={setCurrentAsEnd}
                className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs text-gray-300"
                title="Use current frame"
              >
                Current
              </button>
            </div>
          </div>
        </div>

        <div className="text-xs text-gray-400">
          {hasValidRange ? (
            <span className="text-green-400">{frameCount} frames selected</span>
          ) : (
            <span className="text-red-400">Invalid range</span>
          )}
        </div>
      </div>

      {/* ECG Beat Selection */}
      {hasBeatData && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300">Select from ECG Beats</h4>

          <div className="flex flex-wrap gap-1">
            {beatBoundaries.map((beat, index) => (
              <button
                key={index}
                onClick={() => selectBeat(index)}
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  selectedBeatIndex === index
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Beat {index + 1}
                <span className="text-gray-400 ml-1">
                  ({beat.startFrame}-{beat.endFrame})
                </span>
              </button>
            ))}
          </div>

          {rPeaks && rPeaks.length > 0 && (
            <div className="text-xs text-gray-400">
              {rPeaks.length} R-peaks detected
            </div>
          )}
        </div>
      )}

      {/* Calculate Button */}
      <button
        onClick={handleCalculate}
        disabled={!hasValidRange || isCalculating}
        className={`w-full py-2 rounded font-medium transition-colors ${
          hasValidRange && !isCalculating
            ? 'bg-green-600 hover:bg-green-500 text-white'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
      >
        {isCalculating ? 'Calculating RWS...' : 'Calculate RWS'}
      </button>
    </div>
  );
}
