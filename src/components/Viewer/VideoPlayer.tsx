/**
 * Video Player Component
 *
 * Main viewer that uses the LayerManager canvas system.
 * Features:
 * - 4-layer canvas compositing (Video, Segmentation, Annotation, Overlay)
 * - ECG overlay with timeline scrubbing
 * - Zoom/pan with coordinate transforms
 * - ROI and seed point interactions
 *
 * Refactored to use canvas layer system from Coronary Clear Vision V2.2
 */

import { useCallback, useEffect, useRef, useState, WheelEvent, MouseEvent } from 'react';
import { usePlayerStore } from '../../stores/playerStore';
import { useDicomStore } from '../../stores/dicomStore';
import { useEcgStore } from '../../stores/ecgStore';
import { useSegmentationStore } from '../../stores/segmentationStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { useQcaStore } from '../../stores/qcaStore';
import { useCanvasLayers } from '../../hooks/useCanvasLayers';
import type { EKGData } from '../../lib/canvas';

// Interaction state machine
type InteractionType =
  | { type: 'none' }
  | { type: 'draw-roi' }
  | { type: 'drag-seed'; index: number; offset: { x: number; y: number } }
  | { type: 'drag-roi'; offset: { x: number; y: number } }
  | { type: 'resize-roi'; corner: 'nw' | 'ne' | 'sw' | 'se' }
  | { type: 'pan'; startX: number; startY: number }
  | { type: 'scrub-timeline' };

interface VideoPlayerProps {
  onFrameChange?: (frame: number) => void;
}

export function VideoPlayer({ onFrameChange }: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [interaction, setInteraction] = useState<InteractionType>({ type: 'none' });
  const [cursorStyle, setCursorStyle] = useState('default');
  const lastScrubTime = useRef(0);

  // Store subscriptions
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const metadata = useDicomStore((s) => s.metadata);
  const getFrame = useDicomStore((s) => s.getFrame);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const playbackState = usePlayerStore((s) => s.playbackState);
  const playbackSpeed = usePlayerStore((s) => s.playbackSpeed);
  const frameRate = usePlayerStore((s) => s.frameRate);
  const setCurrentFrame = usePlayerStore((s) => s.setCurrentFrame);
  const nextFrame = usePlayerStore((s) => s.nextFrame);
  const pause = usePlayerStore((s) => s.pause);
  const annotationMode = usePlayerStore((s) => s.annotationMode);

  // ECG store
  const ecgData = useEcgStore((s) => s.ecgData);
  const hasEcg = useEcgStore((s) => s.hasEcg);
  const heartRate = useEcgStore((s) => s.heartRate);

  // Segmentation store - per-frame cached data
  const frameData = useSegmentationStore((s) => s.frameData);
  const getMask = useSegmentationStore((s) => s.getMask);
  const getCenterline = useSegmentationStore((s) => s.getCenterline);

  // Annotation store
  // Subscribe to frameAnnotations Map to trigger re-render when annotations change
  const frameAnnotations = useAnnotationStore((s) => s.frameAnnotations);
  const getSeedPoints = useAnnotationStore((s) => s.getSeedPoints);
  const getROI = useAnnotationStore((s) => s.getROI);
  const addSeedPoint = useAnnotationStore((s) => s.addSeedPoint);
  const updateSeedPoint = useAnnotationStore((s) => s.updateSeedPoint);
  const removeSeedPoint = useAnnotationStore((s) => s.removeSeedPoint);
  const setROI = useAnnotationStore((s) => s.setROI);
  const clearROI = useAnnotationStore((s) => s.clearROI);

  // QCA store
  const getQcaMetrics = useQcaStore((s) => s.getMetrics);

  // Canvas layer system
  const {
    canvasRef,
    isInitialized,
    render,
    updateVideoLayer,
    updateSegmentationLayer,
    updateAnnotationLayer,
    updateOverlayLayer,
    getVideoLayer,
    getOverlayLayer,
    getLayerManager,
  } = useCanvasLayers({ width: dimensions.width, height: dimensions.height });

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Playback loop
  useEffect(() => {
    if (playbackState !== 'playing' || totalFrames === 0) return;

    const interval = setInterval(() => {
      nextFrame();
    }, (1000 / frameRate) / playbackSpeed);

    return () => clearInterval(interval);
  }, [playbackState, totalFrames, frameRate, playbackSpeed, nextFrame]);

  // Notify on frame change
  useEffect(() => {
    onFrameChange?.(currentFrame);
  }, [currentFrame, onFrameChange]);

  // Load and render frame when currentFrame changes
  useEffect(() => {
    if (!isInitialized || !isLoaded) return;

    const loadAndRenderFrame = async () => {
      const frameData = await getFrame(currentFrame);
      if (frameData) {
        await updateVideoLayer({
          frameData: frameData.data,
          width: metadata?.columns || 512,
          height: metadata?.rows || 512,
        });
      }
    };

    loadAndRenderFrame();
  }, [currentFrame, isInitialized, isLoaded, getFrame, updateVideoLayer, metadata]);

  // Update overlay layer with ECG data
  useEffect(() => {
    if (!isInitialized || !hasEcg || !ecgData) return;

    const ekgLayerData: EKGData = {
      signal: ecgData.signal || [],
      time: ecgData.time || [],
      rPeaks: ecgData.rPeaks || [],
      currentFrame: currentFrame,
      totalFrames: totalFrames,
      samplingRate: ecgData.samplingRate,
      heartRate: heartRate ?? ecgData.heartRate ?? undefined,
      duration: ecgData.duration,
    };

    updateOverlayLayer({ ekg: ekgLayerData });
  }, [isInitialized, hasEcg, ecgData, currentFrame, totalFrames, heartRate, updateOverlayLayer]);

  // Update segmentation layer when frame changes - use cached mask
  useEffect(() => {
    if (!isInitialized) return;

    // Get cached mask for current frame
    const cachedMask = getMask(currentFrame);

    // Clear segmentation layer if no mask for this frame
    if (!cachedMask) {
      updateSegmentationLayer({ mask: undefined, color: '#FF0000' });
      return;
    }

    // Convert base64 mask to ImageData
    const loadMaskImage = async () => {
      try {
        const img = new Image();
        img.src = cachedMask;

        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
        });

        // Create canvas to extract ImageData
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        const tempCtx = tempCanvas.getContext('2d');

        if (tempCtx) {
          tempCtx.drawImage(img, 0, 0);
          const imageData = tempCtx.getImageData(0, 0, img.width, img.height);

          updateSegmentationLayer({
            mask: imageData,
            color: '#FF0000', // Bright red for visibility
          });
        }
      } catch (error) {
        console.error('Failed to load mask image:', error);
      }
    };

    loadMaskImage();
  }, [isInitialized, currentFrame, frameData, getMask, updateSegmentationLayer]);

  // Update annotation layer when frame changes
  useEffect(() => {
    if (!isInitialized) return;

    const videoLayer = getVideoLayer();
    if (!videoLayer) return;

    // Get seed points for current frame and convert to canvas coordinates
    const seedPointsImage = getSeedPoints(currentFrame);
    const seedPointsCanvas = seedPointsImage.map((p) => videoLayer.imageToCanvas(p.x, p.y));

    // Get ROI for current frame and convert to canvas coordinates
    const roiImage = getROI(currentFrame);
    let roiCanvas = undefined;
    if (roiImage) {
      const topLeft = videoLayer.imageToCanvas(roiImage.x, roiImage.y);
      const bottomRight = videoLayer.imageToCanvas(
        roiImage.x + roiImage.width,
        roiImage.y + roiImage.height
      );
      roiCanvas = {
        x: topLeft.x,
        y: topLeft.y,
        width: bottomRight.x - topLeft.x,
        height: bottomRight.y - topLeft.y,
      };
    }

    // Get centerline for current frame
    const centerlineImage = getCenterline(currentFrame);
    let centerlines = undefined;
    if (centerlineImage && centerlineImage.length > 0) {
      const centerlineCanvas = centerlineImage.map((p) => videoLayer.imageToCanvas(p.x, p.y));
      centerlines = [{ points: centerlineCanvas, color: '#00FF00', width: 2 }];
    }

    // Get QCA markers
    const qcaMetrics = getQcaMetrics(currentFrame);
    let diameterMarkers = undefined;
    if (qcaMetrics) {
      diameterMarkers = [];
      if (qcaMetrics.mldPosition) {
        const canvasPoint = videoLayer.imageToCanvas(qcaMetrics.mldPosition.x, qcaMetrics.mldPosition.y);
        diameterMarkers.push({
          point: canvasPoint,
          diameter: qcaMetrics.mld,
          type: 'MLD' as const,
        });
      }
      if (qcaMetrics.proximalRdPosition) {
        const canvasPoint = videoLayer.imageToCanvas(qcaMetrics.proximalRdPosition.x, qcaMetrics.proximalRdPosition.y);
        diameterMarkers.push({
          point: canvasPoint,
          diameter: qcaMetrics.proximalRd,
          type: 'Proximal' as const,
        });
      }
      if (qcaMetrics.distalRdPosition) {
        const canvasPoint = videoLayer.imageToCanvas(qcaMetrics.distalRdPosition.x, qcaMetrics.distalRdPosition.y);
        diameterMarkers.push({
          point: canvasPoint,
          diameter: qcaMetrics.distalRd,
          type: 'Distal' as const,
        });
      }
    }

    updateAnnotationLayer({
      seedPoints: seedPointsCanvas,
      roi: roiCanvas,
      centerlines,
      diameterMarkers,
    });
  }, [
    isInitialized,
    currentFrame,
    frameAnnotations, // Re-render when annotations change
    frameData, // Re-render when segmentation data changes (centerlines, etc.)
    getSeedPoints,
    getROI,
    getCenterline,
    getQcaMetrics,
    getVideoLayer,
    updateAnnotationLayer,
  ]);

  // Helper: Convert display coordinates to canvas internal coordinates
  const displayToCanvasCoords = useCallback(
    (displayX: number, displayY: number, canvas: HTMLCanvasElement) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return { x: displayX * scaleX, y: displayY * scaleY };
    },
    []
  );

  // Helper: Check if click is in EKG timeline area
  const isClickInEKGTimeline = useCallback(
    (canvasX: number, canvasY: number): boolean => {
      const overlayLayer = getOverlayLayer();
      return overlayLayer?.isInEKGArea(canvasX, canvasY) ?? false;
    },
    [getOverlayLayer]
  );

  // Helper: Get frame from EKG timeline click
  const getFrameFromEKGClick = useCallback(
    (canvasX: number): number => {
      const overlayLayer = getOverlayLayer();
      return overlayLayer?.getFrameFromClick(canvasX) ?? currentFrame;
    },
    [getOverlayLayer, currentFrame]
  );

  // Helper: Find seed point at position
  const findSeedPointAtPosition = useCallback(
    (canvasX: number, canvasY: number): number | null => {
      const videoLayer = getVideoLayer();
      if (!videoLayer) return null;

      const seedPoints = getSeedPoints(currentFrame);
      const hitRadius = 15;

      for (let i = 0; i < seedPoints.length; i++) {
        const canvasPoint = videoLayer.imageToCanvas(seedPoints[i].x, seedPoints[i].y);
        const dx = canvasX - canvasPoint.x;
        const dy = canvasY - canvasPoint.y;
        if (Math.sqrt(dx * dx + dy * dy) <= hitRadius) {
          return i;
        }
      }
      return null;
    },
    [getVideoLayer, getSeedPoints, currentFrame]
  );

  // Handle wheel zoom
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const videoLayer = getVideoLayer();
      if (!videoLayer) return;

      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const displayX = e.clientX - rect.left;
      const displayY = e.clientY - rect.top;
      const { x: canvasX, y: canvasY } = displayToCanvasCoords(
        displayX,
        displayY,
        e.target as HTMLCanvasElement
      );

      const zoomFactor = 1.1;
      if (e.deltaY < 0) {
        videoLayer.zoomIn(zoomFactor, canvasX, canvasY);
      } else {
        videoLayer.zoomOut(zoomFactor, canvasX, canvasY);
      }

      render();
    },
    [getVideoLayer, displayToCanvasCoords, render]
  );

  // Handle mouse down
  const handleMouseDown = useCallback(
    (e: MouseEvent) => {
      const canvas = e.target as HTMLCanvasElement;
      const rect = canvas.getBoundingClientRect();
      const displayX = e.clientX - rect.left;
      const displayY = e.clientY - rect.top;
      const { x: canvasX, y: canvasY } = displayToCanvasCoords(displayX, displayY, canvas);

      const videoLayer = getVideoLayer();

      // Priority 0: EKG timeline scrubbing
      if (isClickInEKGTimeline(canvasX, canvasY) && annotationMode === 'none') {
        const frame = getFrameFromEKGClick(canvasX);
        pause();
        setCurrentFrame(frame);
        setInteraction({ type: 'scrub-timeline' });
        return;
      }

      // Priority 1: Pan mode (middle mouse or Alt+left click)
      if (e.button === 1 || (e.button === 0 && e.altKey) || annotationMode === 'pan') {
        e.preventDefault();
        setInteraction({ type: 'pan', startX: canvasX, startY: canvasY });
        setCursorStyle('grabbing');
        return;
      }

      // Priority 2: Seed point drag
      const seedIndex = findSeedPointAtPosition(canvasX, canvasY);
      if (seedIndex !== null && videoLayer) {
        const seedPoints = getSeedPoints(currentFrame);
        const canvasPoint = videoLayer.imageToCanvas(seedPoints[seedIndex].x, seedPoints[seedIndex].y);
        setInteraction({
          type: 'drag-seed',
          index: seedIndex,
          offset: { x: canvasX - canvasPoint.x, y: canvasY - canvasPoint.y },
        });
        return;
      }

      // Priority 3: Seed mode - add new seed point
      if (annotationMode === 'seed' && videoLayer) {
        const imagePoint = videoLayer.canvasToImage(canvasX, canvasY);
        addSeedPoint(currentFrame, imagePoint);
        return;
      }

      // Priority 4: ROI mode - start drawing
      if (annotationMode === 'roi' && videoLayer) {
        const imagePoint = videoLayer.canvasToImage(canvasX, canvasY);
        setROI(currentFrame, { x: imagePoint.x, y: imagePoint.y, width: 0, height: 0 });
        setInteraction({ type: 'draw-roi' });
        return;
      }
    },
    [
      displayToCanvasCoords,
      getVideoLayer,
      isClickInEKGTimeline,
      getFrameFromEKGClick,
      findSeedPointAtPosition,
      annotationMode,
      currentFrame,
      getSeedPoints,
      addSeedPoint,
      setROI,
      pause,
      setCurrentFrame,
    ]
  );

  // Handle mouse move
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      const canvas = e.target as HTMLCanvasElement;
      const rect = canvas.getBoundingClientRect();
      const displayX = e.clientX - rect.left;
      const displayY = e.clientY - rect.top;
      const { x: canvasX, y: canvasY } = displayToCanvasCoords(displayX, displayY, canvas);

      const videoLayer = getVideoLayer();
      const layerManager = getLayerManager();

      // Handle active interactions
      if (interaction.type === 'scrub-timeline') {
        const now = Date.now();
        if (now - lastScrubTime.current > 32) {
          const frame = getFrameFromEKGClick(canvasX);
          setCurrentFrame(frame);
          lastScrubTime.current = now;
        }
        return;
      }

      if (interaction.type === 'pan' && videoLayer) {
        const deltaX = canvasX - interaction.startX;
        const deltaY = canvasY - interaction.startY;
        videoLayer.pan(deltaX, deltaY);
        layerManager?.render();
        setInteraction({ type: 'pan', startX: canvasX, startY: canvasY });
        return;
      }

      if (interaction.type === 'drag-seed' && videoLayer) {
        const imagePoint = videoLayer.canvasToImage(canvasX, canvasY);
        updateSeedPoint(currentFrame, interaction.index, imagePoint);
        return;
      }

      if (interaction.type === 'draw-roi' && videoLayer) {
        const roi = getROI(currentFrame);
        if (roi) {
          const imagePoint = videoLayer.canvasToImage(canvasX, canvasY);
          const newWidth = imagePoint.x - roi.x;
          const newHeight = imagePoint.y - roi.y;
          setROI(currentFrame, { ...roi, width: newWidth, height: newHeight });
        }
        return;
      }

      // Update cursor based on hover
      let newCursor = 'default';

      if (isClickInEKGTimeline(canvasX, canvasY) && annotationMode === 'none') {
        newCursor = 'pointer';
      } else if (annotationMode === 'pan') {
        newCursor = 'grab';
      } else if (annotationMode === 'seed') {
        newCursor = 'crosshair';
      } else if (annotationMode === 'roi') {
        newCursor = 'crosshair';
      } else if (findSeedPointAtPosition(canvasX, canvasY) !== null) {
        newCursor = 'grab';
      }

      setCursorStyle(newCursor);
    },
    [
      displayToCanvasCoords,
      getVideoLayer,
      getLayerManager,
      interaction,
      isClickInEKGTimeline,
      getFrameFromEKGClick,
      findSeedPointAtPosition,
      annotationMode,
      currentFrame,
      updateSeedPoint,
      getROI,
      setROI,
      setCurrentFrame,
    ]
  );

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    if (interaction.type === 'draw-roi') {
      // Normalize ROI (ensure positive width/height)
      const roi = getROI(currentFrame);
      if (roi) {
        const normalized = {
          x: roi.width < 0 ? roi.x + roi.width : roi.x,
          y: roi.height < 0 ? roi.y + roi.height : roi.y,
          width: Math.abs(roi.width),
          height: Math.abs(roi.height),
        };
        if (normalized.width > 10 && normalized.height > 10) {
          setROI(currentFrame, normalized);
        } else {
          clearROI(currentFrame); // Too small, remove
        }
      }
    }

    setInteraction({ type: 'none' });
    setCursorStyle('default');
  }, [interaction, currentFrame, getROI, setROI, clearROI]);

  // Handle right-click to remove seed point
  const handleContextMenu = useCallback(
    (e: MouseEvent) => {
      e.preventDefault();

      const canvas = e.target as HTMLCanvasElement;
      const rect = canvas.getBoundingClientRect();
      const displayX = e.clientX - rect.left;
      const displayY = e.clientY - rect.top;
      const { x: canvasX, y: canvasY } = displayToCanvasCoords(displayX, displayY, canvas);

      // Remove seed point if clicking on one
      const seedIndex = findSeedPointAtPosition(canvasX, canvasY);
      if (seedIndex !== null) {
        removeSeedPoint(currentFrame, seedIndex);
      }
    },
    [displayToCanvasCoords, findSeedPointAtPosition, removeSeedPoint, currentFrame]
  );

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        const videoLayer = getVideoLayer();
        if (videoLayer) {
          videoLayer.resetView();
          render();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [getVideoLayer, render]);

  // Empty state
  if (!isLoaded) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black text-gray-500">
        <div className="text-center">
          <p className="text-xl mb-2">Coronary RWS Analyser</p>
          <p className="text-sm">Open a DICOM file to begin analysis</p>
          <p className="text-xs mt-4 text-gray-600">
            Drag and drop or use File menu
          </p>
        </div>
      </div>
    );
  }

  const videoLayer = getVideoLayer();
  const zoomLevel = videoLayer?.getZoomLevel() ?? 1;

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative bg-black overflow-hidden"
      style={{ cursor: cursorStyle }}
    >
      {/* Single composite canvas managed by LayerManager */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className="absolute inset-0 w-full h-full"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleContextMenu}
      />

      {/* Frame info overlay */}
      <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-xs text-white pointer-events-none">
        Frame: {currentFrame + 1} / {totalFrames}
        {metadata && (
          <span className="ml-2 text-gray-400">
            ({metadata.rows}×{metadata.columns})
          </span>
        )}
      </div>

      {/* Zoom indicator */}
      <div className="absolute top-2 right-2 bg-black/50 px-2 py-1 rounded text-xs text-white pointer-events-none">
        <span className="text-gray-400">Zoom:</span> {Math.round(zoomLevel * 100)}%
      </div>

      {/* Mode indicator */}
      {annotationMode !== 'none' && (
        <div className="absolute top-10 right-2 bg-blue-500/80 px-2 py-1 rounded text-xs text-white pointer-events-none">
          Mode: {annotationMode.toUpperCase()}
        </div>
      )}

      {/* Help hint */}
      <div className="absolute bottom-2 right-2 bg-black/50 px-2 py-1 rounded text-xs text-gray-400 pointer-events-none">
        Scroll: Zoom | Alt+drag: Pan | R: Reset | Right-click: Remove seed
      </div>
    </div>
  );
}
