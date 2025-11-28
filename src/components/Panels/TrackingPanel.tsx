/**
 * Tracking Panel Component
 *
 * CSRT-based ROI tracking controls with WebSocket real-time updates.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useTrackingStore, usePlayerStore, useDicomStore, useAnnotationStore } from '../../stores';
import { api } from '../../lib/api';

type TrackingDirection = 'forward' | 'backward';

export function TrackingPanel() {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const setCurrentFrame = usePlayerStore((s) => s.setCurrentFrame);
  const isLoaded = useDicomStore((s) => s.isLoaded);

  // Get ROI from annotation store
  const frameAnnotations = useAnnotationStore((s) => s.frameAnnotations);
  const setROI = useAnnotationStore((s) => s.setROI);
  const currentRoi = frameAnnotations.get(currentFrame)?.roi ?? null;

  const isPropagating = useTrackingStore((s) => s.isPropagating);
  const propagationProgress = useTrackingStore((s) => s.propagationProgress);
  const setIsPropagating = useTrackingStore((s) => s.setIsPropagating);
  const setPropagationProgress = useTrackingStore((s) => s.setPropagationProgress);

  const [direction, setDirection] = useState<TrackingDirection>('forward');
  const [endFrame, setEndFrame] = useState(totalFrames > 0 ? totalFrames - 1 : 0);

  // Update endFrame when direction changes
  useEffect(() => {
    if (direction === 'forward') {
      setEndFrame(totalFrames > 0 ? totalFrames - 1 : 0);
    } else {
      setEndFrame(0); // Backward: go to frame 0
    }
  }, [direction, totalFrames]);
  const [error, setError] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [isInitialized, setIsInitialized] = useState(false);
  const [trackingResults, setTrackingResults] = useState<{
    framesTracked: number;
    avgConfidence: number;
    stoppedEarly: boolean;
    stopReason?: string;
  } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // Initialize CSRT tracker (ROI only)
  const handleInitialize = useCallback(async () => {
    if (!currentRoi) {
      setError('Draw ROI first (press B key)');
      return;
    }

    setError(null);

    try {
      const response = await api.tracking.initialize({
        frameIndex: currentFrame,
        roi: currentRoi,
      });

      if (response.status === 'initialized') {
        setIsInitialized(true);
        setTrackingResults(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to initialize tracker');
    }
  }, [currentFrame, currentRoi]);

  // Start propagation with WebSocket
  const handlePropagate = useCallback(async () => {
    if (!isInitialized) {
      setError('Initialize tracker first');
      return;
    }

    setError(null);
    setTrackingResults(null);
    setIsPropagating(true);
    setPropagationProgress({ progress: 0, isRunning: true, status: 'running' });

    // Connect to WebSocket
    setWsStatus('connecting');

    try {
      const ws = new WebSocket('ws://127.0.0.1:8000/tracking/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus('connected');

        // Send propagate command
        ws.send(JSON.stringify({
          action: 'propagate',
          start_frame: currentFrame,
          end_frame: endFrame,
          direction: direction,
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'progress':
            setPropagationProgress({ progress: data.percent, currentFrame: data.current_frame });
            break;

          case 'frame_update':
            // Live update: show tracked ROI on each frame
            if (data.success && data.roi) {
              // Update current frame (like playback)
              setCurrentFrame(data.frame_index);
              // Save ROI to annotation store (so it's displayed on canvas)
              setROI(data.frame_index, {
                x: data.roi.x,
                y: data.roi.y,
                width: data.roi.width,
                height: data.roi.height,
              });
            }
            break;

          case 'completed':
            setTrackingResults({
              framesTracked: data.num_frames_tracked,
              avgConfidence: data.results?.reduce((sum: number, r: any) => sum + r.confidence, 0) / (data.num_frames_tracked || 1),
              stoppedEarly: data.stopped_early,
              stopReason: data.stop_reason,
            });
            setIsPropagating(false);
            setPropagationProgress({ progress: 100, isRunning: false, status: data.stopped_early ? 'stopped' : 'completed' });
            ws.close();
            break;

          case 'error':
            setError(data.message);
            setIsPropagating(false);
            ws.close();
            break;
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection error');
        setWsStatus('disconnected');
        setIsPropagating(false);
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        wsRef.current = null;
      };

    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start propagation');
      setIsPropagating(false);
    }
  }, [currentFrame, endFrame, direction, isInitialized, setIsPropagating, setPropagationProgress, setCurrentFrame, setROI]);

  // Stop propagation
  const handleStop = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
  }, []);

  // Reset tracker
  const handleReset = useCallback(async () => {
    try {
      await api.tracking.reset();
      setIsInitialized(false);
      setTrackingResults(null);
      setError(null);
      setPropagationProgress({ progress: 0, isRunning: false, status: 'idle' });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reset tracker');
    }
  }, [setPropagationProgress]);

  const canInitialize = isLoaded && currentRoi !== null;
  const canPropagate = isLoaded && isInitialized && !isPropagating;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-2">
        ROI Tracking (CSRT)
      </h3>

      {/* Status */}
      <div className="bg-gray-800 rounded p-3 space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-400">ROI:</span>
          <span className={currentRoi ? 'text-green-400' : 'text-gray-500'}>
            {currentRoi ? `${currentRoi.width}x${currentRoi.height}` : 'Not defined (press B)'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Tracker:</span>
          <span className={isInitialized ? 'text-green-400' : 'text-gray-500'}>
            {isInitialized ? 'Initialized' : 'Not initialized'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">WebSocket:</span>
          <span className={
            wsStatus === 'connected' ? 'text-green-400' :
            wsStatus === 'connecting' ? 'text-yellow-400' : 'text-gray-500'
          }>
            {wsStatus}
          </span>
        </div>
      </div>

      {/* Initialize */}
      <button
        onClick={handleInitialize}
        disabled={!canInitialize || isPropagating}
        className={`w-full py-2 rounded text-sm transition-colors ${
          canInitialize && !isPropagating
            ? 'bg-blue-600 hover:bg-blue-500 text-white'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
      >
        Initialize Tracker
      </button>

      {/* Propagation Controls */}
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Direction</label>
            <select
              value={direction}
              onChange={(e) => setDirection(e.target.value as TrackingDirection)}
              disabled={isPropagating}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white"
            >
              <option value="forward">Forward</option>
              <option value="backward">Backward</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">End Frame</label>
            <input
              type="number"
              value={endFrame}
              onChange={(e) => setEndFrame(parseInt(e.target.value) || 0)}
              min={0}
              max={totalFrames - 1}
              disabled={isPropagating}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white"
            />
          </div>
        </div>

        {/* Progress Bar */}
        {isPropagating && (
          <div className="space-y-1">
            <div className="h-2 bg-gray-700 rounded overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all duration-200"
                style={{ width: `${propagationProgress.progress}%` }}
              />
            </div>
            <div className="text-xs text-gray-400 text-center">
              {propagationProgress.progress.toFixed(0)}%
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-2">
          {isPropagating ? (
            <button
              onClick={handleStop}
              className="col-span-2 py-2 bg-red-600 hover:bg-red-500 text-white rounded text-sm transition-colors"
            >
              Stop
            </button>
          ) : (
            <>
              <button
                onClick={handlePropagate}
                disabled={!canPropagate}
                className={`py-2 rounded text-sm transition-colors ${
                  canPropagate
                    ? 'bg-green-600 hover:bg-green-500 text-white'
                    : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
              >
                Propagate
              </button>
              <button
                onClick={handleReset}
                className="py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
              >
                Reset
              </button>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Results */}
      {trackingResults && (
        <div className={`rounded p-3 text-xs space-y-1 ${
          trackingResults.stoppedEarly
            ? 'bg-yellow-900/30 border border-yellow-700'
            : 'bg-green-900/30 border border-green-700'
        }`}>
          <div className="flex justify-between">
            <span className="text-gray-400">Frames Tracked:</span>
            <span className="text-white">{trackingResults.framesTracked}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Avg Confidence:</span>
            <span className={
              trackingResults.avgConfidence > 0.7 ? 'text-green-400' :
              trackingResults.avgConfidence > 0.5 ? 'text-yellow-400' : 'text-red-400'
            }>
              {(trackingResults.avgConfidence * 100).toFixed(1)}%
            </span>
          </div>
          {trackingResults.stoppedEarly && trackingResults.stopReason && (
            <div className="text-yellow-400 mt-1">
              Stopped: {trackingResults.stopReason}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
