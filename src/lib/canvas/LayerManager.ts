/**
 * Layer Manager
 * Manages and composites multiple canvas layers
 *
 * Features:
 * - 4-layer rendering system (Video, Segmentation, Annotation, Overlay)
 * - Z-index based compositing
 * - Quality level support for performance
 * - Dirty tracking for efficient re-renders
 * - Performance metrics
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { Layer } from './Layer';
import { VideoLayer } from './VideoLayer';
import { SegmentationLayer } from './SegmentationLayer';
import { AnnotationLayer } from './AnnotationLayer';
import { OverlayLayer } from './OverlayLayer';
import {
  LayerType,
  RenderingMode,
  QualityLevel,
  LayerRenderOptions,
  PerformanceMetrics,
  LayerManagerInterface,
} from './types';

export class LayerManager implements LayerManagerInterface {
  private layers: Map<LayerType, Layer> = new Map();
  private compositeCanvas: HTMLCanvasElement;
  private compositeCtx: CanvasRenderingContext2D;
  private renderingMode: RenderingMode = RenderingMode.CANVAS2D;
  private qualityLevel: QualityLevel = QualityLevel.LOW;
  private performanceMetrics: PerformanceMetrics = {
    fps: 0,
    renderTime: 0,
    layerCount: 0,
    mode: RenderingMode.CANVAS2D,
  };

  constructor(canvas: HTMLCanvasElement) {
    this.compositeCanvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('Failed to create composite canvas context');
    this.compositeCtx = ctx;

    // Detect and set best rendering mode
    this.renderingMode = this.detectRenderingMode();

    // Initialize layers
    this.initializeLayers();
  }

  /**
   * Detect best available rendering mode
   */
  private detectRenderingMode(): RenderingMode {
    const tempCanvas = document.createElement('canvas');

    // Try WebGL2
    const webgl2 = tempCanvas.getContext('webgl2');
    if (webgl2) return RenderingMode.WEBGL2;

    // Try WebGL
    const webgl = tempCanvas.getContext('webgl');
    if (webgl) return RenderingMode.WEBGL;

    // Fallback to Canvas 2D
    return RenderingMode.CANVAS2D;
  }

  /**
   * Initialize all layer types
   */
  private initializeLayers(): void {
    this.layers.set(LayerType.VIDEO, new VideoLayer());
    this.layers.set(LayerType.SEGMENTATION, new SegmentationLayer());
    this.layers.set(LayerType.ANNOTATION, new AnnotationLayer());
    this.layers.set(LayerType.OVERLAY, new OverlayLayer());

    // Resize all layers to match composite canvas
    this.resizeLayers(this.compositeCanvas.width, this.compositeCanvas.height);
  }

  /**
   * Get layer by type
   */
  getLayer<T>(type: LayerType): T | undefined {
    return this.layers.get(type) as T | undefined;
  }

  /**
   * Resize all layers
   */
  resize(width: number, height: number): void {
    if (
      this.compositeCanvas.width !== width ||
      this.compositeCanvas.height !== height
    ) {
      this.compositeCanvas.width = width;
      this.compositeCanvas.height = height;
      this.resizeLayers(width, height);
    }
  }

  /**
   * Resize all layers to match size
   */
  private resizeLayers(width: number, height: number): void {
    this.layers.forEach((layer) => {
      layer.resize(width, height);
    });
  }

  /**
   * Set quality level (affects overlay opacity)
   */
  setQualityLevel(level: QualityLevel): void {
    if (this.qualityLevel !== level) {
      this.qualityLevel = level;
      // Mark all layers as dirty to trigger re-render with new quality
      this.layers.forEach((layer) => layer.markDirty());
    }
  }

  /**
   * Get current quality level
   */
  getQualityLevel(): QualityLevel {
    return this.qualityLevel;
  }

  /**
   * Render all layers and composite to main canvas
   */
  render(timestamp?: number): void {
    const startTime = performance.now();

    // Clear composite canvas
    this.compositeCtx.fillStyle = '#000000';
    this.compositeCtx.fillRect(
      0,
      0,
      this.compositeCanvas.width,
      this.compositeCanvas.height
    );

    // Render options
    const options: LayerRenderOptions = {
      quality: this.qualityLevel,
      timestamp,
      layerManager: this, // Pass LayerManager reference for inter-layer access
    };

    // Get layers sorted by z-index
    const sortedLayers = this.getSortedLayers();

    // Render each layer
    sortedLayers.forEach((layer) => {
      // Always render all layers to ensure zoom/pan sync
      // Each layer gets transform from VideoLayer for proper positioning
      const shouldAlwaysRender = true;

      if (layer.isVisible() && (layer.isDirty() || shouldAlwaysRender)) {
        layer.render(options);
      }

      // Composite layer to main canvas if visible
      if (layer.isVisible()) {
        const layerCanvas = layer.getCanvas();
        const config = layer.getConfig();

        this.compositeCtx.save();
        this.compositeCtx.globalAlpha = config.opacity;

        if (config.blendMode) {
          this.compositeCtx.globalCompositeOperation = config.blendMode;
        }

        this.compositeCtx.drawImage(layerCanvas, 0, 0);
        this.compositeCtx.restore();
      }
    });

    // Update performance metrics
    const renderTime = performance.now() - startTime;
    this.updatePerformanceMetrics(renderTime);
  }

  /**
   * Get layers sorted by z-index
   */
  private getSortedLayers(): Layer[] {
    return Array.from(this.layers.values()).sort((a, b) => {
      return a.getConfig().zIndex - b.getConfig().zIndex;
    });
  }

  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(renderTime: number): void {
    this.performanceMetrics.renderTime = renderTime;
    this.performanceMetrics.fps = renderTime > 0 ? 1000 / renderTime : 0;
    this.performanceMetrics.layerCount = this.layers.size;
    this.performanceMetrics.mode = this.renderingMode;
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  /**
   * Set layer visibility
   */
  setLayerVisibility(type: LayerType, visible: boolean): void {
    const layer = this.layers.get(type);
    if (layer) {
      layer.setVisible(visible);
    }
  }

  /**
   * Set layer opacity
   */
  setLayerOpacity(type: LayerType, opacity: number): void {
    const layer = this.layers.get(type);
    if (layer) {
      layer.setOpacity(opacity);
    }
  }

  /**
   * Toggle layer visibility
   */
  toggleLayerVisibility(type: LayerType): void {
    const layer = this.layers.get(type);
    if (layer) {
      layer.setVisible(!layer.isVisible());
    }
  }

  /**
   * Mark all layers as dirty (force re-render)
   */
  markAllDirty(): void {
    this.layers.forEach((layer) => layer.markDirty());
  }

  /**
   * Get rendering mode
   */
  getRenderingMode(): RenderingMode {
    return this.renderingMode;
  }

  /**
   * Get composite canvas
   */
  getCanvas(): HTMLCanvasElement {
    return this.compositeCanvas;
  }

  /**
   * Get canvas dimensions
   */
  getDimensions(): { width: number; height: number } {
    return {
      width: this.compositeCanvas.width,
      height: this.compositeCanvas.height,
    };
  }

  /**
   * Dispose all resources
   */
  dispose(): void {
    this.layers.forEach((layer) => layer.dispose());
    this.layers.clear();
  }
}
