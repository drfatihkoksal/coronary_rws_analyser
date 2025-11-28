/**
 * Playback Controls Component
 *
 * Video playback controls with play/pause, frame navigation, speed, and timeline.
 */

import { useCallback } from 'react';
import { usePlayerStore, useDicomStore } from '../../stores';
import { Button } from '../common/Button';
import { Slider } from '../common/Slider';

export function PlaybackControls() {
  const isLoaded = useDicomStore((s) => s.isLoaded);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const playbackState = usePlayerStore((s) => s.playbackState);
  const playbackSpeed = usePlayerStore((s) => s.playbackSpeed);
  const isLooping = usePlayerStore((s) => s.isLooping);
  const frameRate = usePlayerStore((s) => s.frameRate);

  const setCurrentFrame = usePlayerStore((s) => s.setCurrentFrame);
  const nextFrame = usePlayerStore((s) => s.nextFrame);
  const previousFrame = usePlayerStore((s) => s.previousFrame);
  const togglePlayback = usePlayerStore((s) => s.togglePlayback);
  const setPlaybackSpeed = usePlayerStore((s) => s.setPlaybackSpeed);
  const setLooping = usePlayerStore((s) => s.setLooping);

  // Handle timeline change
  const handleTimelineChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setCurrentFrame(parseInt(e.target.value, 10));
    },
    [setCurrentFrame]
  );

  // Speed options
  const speeds = [0.25, 0.5, 1, 2];

  if (!isLoaded) {
    return (
      <div className="h-full flex flex-col p-2 opacity-50">
        <div className="flex-1 bg-gray-700 rounded mb-2" />
        <div className="flex items-center justify-center gap-4">
          <Button disabled size="sm">⏮</Button>
          <Button disabled variant="primary" size="sm">▶</Button>
          <Button disabled size="sm">⏭</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-2">
      {/* Timeline slider */}
      <div className="flex-1 flex flex-col justify-center mb-2">
        <Slider
          min={0}
          max={Math.max(0, totalFrames - 1)}
          value={currentFrame}
          onChange={handleTimelineChange}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0</span>
          <span>{Math.round(currentFrame / frameRate * 10) / 10}s</span>
          <span>{totalFrames - 1}</span>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-center gap-2">
        {/* Navigation */}
        <Button onClick={previousFrame} size="sm" title="Previous frame (←)">
          ⏮
        </Button>

        <Button
          onClick={togglePlayback}
          variant="primary"
          size="sm"
          className="w-16"
          title="Play/Pause (Space)"
        >
          {playbackState === 'playing' ? '⏸' : '▶'}
        </Button>

        <Button onClick={nextFrame} size="sm" title="Next frame (→)">
          ⏭
        </Button>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-600 mx-2" />

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-400">Speed:</span>
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="bg-gray-700 text-white text-xs rounded px-1 py-0.5 border border-gray-600"
          >
            {speeds.map((s) => (
              <option key={s} value={s}>
                {s}x
              </option>
            ))}
          </select>
        </div>

        {/* Loop toggle */}
        <Button
          onClick={() => setLooping(!isLooping)}
          variant={isLooping ? 'primary' : 'ghost'}
          size="sm"
          title="Toggle loop"
        >
          🔁
        </Button>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-600 mx-2" />

        {/* Frame info */}
        <span className="text-sm text-gray-300 min-w-[100px]">
          Frame: <span className="font-mono">{currentFrame + 1}</span> / {totalFrames}
        </span>

        {/* FPS */}
        <span className="text-xs text-gray-500">
          ({frameRate} FPS)
        </span>
      </div>
    </div>
  );
}
