/**
 * useCanvasLayers Hook
 * React hook for managing canvas layer system
 *
 * Provides:
 * - Canvas initialization via callback ref
 * - Layer update methods
 * - Render loop management
 * - Layer visibility controls
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  LayerManager,
  LayerType,
  QualityLevel,
  VideoLayer,
  SegmentationLayer,
  AnnotationLayer,
  OverlayLayer,
} from '../lib/canvas';
import type {
  VideoLayerData,
  SegmentationData,
  AnnotationData,
  OverlayData,
  PerformanceMetrics,
} from '../lib/canvas';

export interface UseCanvasLayersOptions {
  width: number;
  height: number;
}

export function useCanvasLayers(options: UseCanvasLayersOptions) {
  const layerManagerRef = useRef<LayerManager | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [isInitialized, setIsInitialized] = useState(false);
  const [metrics] = useState<PerformanceMetrics | null>(null);

  /**
   * Callback ref for canvas initialization
   * This is called when canvas is mounted to DOM
   */
  const canvasRef = useCallback((node: HTMLCanvasElement | null) => {
    if (node && !layerManagerRef.current) {
      layerManagerRef.current = new LayerManager(node);
      setIsInitialized(true);
    } else if (!node && layerManagerRef.current) {
      layerManagerRef.current.dispose();
      layerManagerRef.current = null;
      setIsInitialized(false);

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }
  }, []);

  /**
   * Handle canvas resize
   */
  useEffect(() => {
    if (layerManagerRef.current) {
      layerManagerRef.current.resize(options.width, options.height);
      // Render after resize to ensure layers are properly drawn
      layerManagerRef.current.render();
    }
  }, [options.width, options.height]);

  /**
   * Handle quality level - use LOW for consistent appearance
   */
  useEffect(() => {
    if (layerManagerRef.current) {
      layerManagerRef.current.setQualityLevel(QualityLevel.LOW);
    }
  }, []);

  /**
   * Render single frame
   */
  const render = useCallback((timestamp?: number) => {
    if (layerManagerRef.current) {
      layerManagerRef.current.render(timestamp);
    }
  }, []);

  /**
   * Start continuous rendering (for playback)
   */
  const startRenderLoop = useCallback(() => {
    const loop = (timestamp: number) => {
      render(timestamp);
      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);
  }, [render]);

  /**
   * Stop continuous rendering
   */
  const stopRenderLoop = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  /**
   * Update video layer with frame data
   */
  const updateVideoLayer = useCallback(
    async (data: VideoLayerData) => {
      if (!layerManagerRef.current) return;

      const videoLayer = layerManagerRef.current.getLayer<VideoLayer>(
        LayerType.VIDEO
      );

      if (videoLayer) {
        await videoLayer.loadFrame(data);
        render(); // Single render for frame change
      }
    },
    [render]
  );

  /**
   * Update segmentation layer
   */
  const updateSegmentationLayer = useCallback(
    (data: SegmentationData) => {
      if (!layerManagerRef.current) return;

      const segLayer = layerManagerRef.current.getLayer<SegmentationLayer>(
        LayerType.SEGMENTATION
      );

      if (segLayer) {
        segLayer.updateData(data);
        render();
      }
    },
    [render]
  );

  /**
   * Clear segmentation layer
   */
  const clearSegmentationLayer = useCallback(() => {
    if (!layerManagerRef.current) return;

    const segLayer = layerManagerRef.current.getLayer<SegmentationLayer>(
      LayerType.SEGMENTATION
    );

    if (segLayer) {
      segLayer.clearData();
      render();
    }
  }, [render]);

  /**
   * Update annotation layer
   */
  const updateAnnotationLayer = useCallback(
    (data: Partial<AnnotationData>) => {
      if (!layerManagerRef.current) return;

      const annotationLayer = layerManagerRef.current.getLayer<AnnotationLayer>(
        LayerType.ANNOTATION
      );

      if (annotationLayer) {
        annotationLayer.updateData(data);
        render();
      }
    },
    [render]
  );

  /**
   * Clear annotation layer
   */
  const clearAnnotationLayer = useCallback(() => {
    if (!layerManagerRef.current) return;

    const annotationLayer = layerManagerRef.current.getLayer<AnnotationLayer>(
      LayerType.ANNOTATION
    );

    if (annotationLayer) {
      annotationLayer.clearData();
      render();
    }
  }, [render]);

  /**
   * Update overlay layer
   */
  const updateOverlayLayer = useCallback(
    (data: Partial<OverlayData>) => {
      if (!layerManagerRef.current) return;

      const overlayLayer = layerManagerRef.current.getLayer<OverlayLayer>(
        LayerType.OVERLAY
      );

      if (overlayLayer) {
        overlayLayer.updateData(data);
        render();
      }
    },
    [render]
  );

  /**
   * Update overlay current frame (for ECG scrubber)
   */
  const setOverlayCurrentFrame = useCallback(
    (frame: number) => {
      if (!layerManagerRef.current) return;

      const overlayLayer = layerManagerRef.current.getLayer<OverlayLayer>(
        LayerType.OVERLAY
      );

      if (overlayLayer) {
        overlayLayer.setCurrentFrame(frame);
        render();
      }
    },
    [render]
  );

  /**
   * Set layer visibility
   */
  const setLayerVisibility = useCallback(
    (type: LayerType, visible: boolean) => {
      if (layerManagerRef.current) {
        layerManagerRef.current.setLayerVisibility(type, visible);
        render();
      }
    },
    [render]
  );

  /**
   * Toggle layer visibility
   */
  const toggleLayerVisibility = useCallback(
    (type: LayerType) => {
      if (layerManagerRef.current) {
        layerManagerRef.current.toggleLayerVisibility(type);
        render();
      }
    },
    [render]
  );

  /**
   * Set layer opacity
   */
  const setLayerOpacity = useCallback(
    (type: LayerType, opacity: number) => {
      if (layerManagerRef.current) {
        layerManagerRef.current.setLayerOpacity(type, opacity);
        render();
      }
    },
    [render]
  );

  /**
   * Set quality level
   */
  const setQualityLevel = useCallback(
    (quality: QualityLevel) => {
      if (layerManagerRef.current) {
        layerManagerRef.current.setQualityLevel(quality);
        render();
      }
    },
    [render]
  );

  /**
   * Get layer manager (for advanced operations)
   */
  const getLayerManager = useCallback((): LayerManager | null => {
    return layerManagerRef.current;
  }, []);

  /**
   * Get video layer (for coordinate transforms)
   */
  const getVideoLayer = useCallback((): VideoLayer | null => {
    if (!layerManagerRef.current) return null;
    return layerManagerRef.current.getLayer<VideoLayer>(LayerType.VIDEO) || null;
  }, []);

  /**
   * Get overlay layer (for EKG interaction)
   */
  const getOverlayLayer = useCallback((): OverlayLayer | null => {
    if (!layerManagerRef.current) return null;
    return layerManagerRef.current.getLayer<OverlayLayer>(LayerType.OVERLAY) || null;
  }, []);

  /**
   * Get annotation layer
   */
  const getAnnotationLayer = useCallback((): AnnotationLayer | null => {
    if (!layerManagerRef.current) return null;
    return layerManagerRef.current.getLayer<AnnotationLayer>(LayerType.ANNOTATION) || null;
  }, []);

  /**
   * Set annotation visibility flags (for individual overlays within annotation layer)
   */
  const setAnnotationVisibility = useCallback(
    (flags: {
      roi?: boolean;
      centerline?: boolean;
      seedPoints?: boolean;
      yoloKeypoints?: boolean;
      diameterMarkers?: boolean;
    }) => {
      if (!layerManagerRef.current) return;

      const annotationLayer = layerManagerRef.current.getLayer<AnnotationLayer>(
        LayerType.ANNOTATION
      );

      if (annotationLayer) {
        annotationLayer.setVisibility(flags);
        render();
      }
    },
    [render]
  );

  /**
   * Reset view (zoom/pan)
   */
  const resetView = useCallback(() => {
    if (!layerManagerRef.current) return;

    const videoLayer = layerManagerRef.current.getLayer<VideoLayer>(
      LayerType.VIDEO
    );

    if (videoLayer) {
      videoLayer.resetView();
      render();
    }
  }, [render]);

  /**
   * Mark all layers dirty (force re-render)
   */
  const markAllDirty = useCallback(() => {
    if (layerManagerRef.current) {
      layerManagerRef.current.markAllDirty();
      render();
    }
  }, [render]);

  return {
    // Refs
    canvasRef,

    // State
    isInitialized,
    metrics,

    // Render
    render,
    startRenderLoop,
    stopRenderLoop,

    // Layer updates
    updateVideoLayer,
    updateSegmentationLayer,
    clearSegmentationLayer,
    updateAnnotationLayer,
    clearAnnotationLayer,
    updateOverlayLayer,
    setOverlayCurrentFrame,

    // Visibility controls
    setLayerVisibility,
    toggleLayerVisibility,
    setLayerOpacity,
    setQualityLevel,

    // Layer access
    getLayerManager,
    getVideoLayer,
    getOverlayLayer,
    getAnnotationLayer,

    // Annotation visibility
    setAnnotationVisibility,

    // Utilities
    resetView,
    markAllDirty,
  };
}
