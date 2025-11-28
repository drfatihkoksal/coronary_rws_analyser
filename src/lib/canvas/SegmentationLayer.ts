/**
 * Segmentation Layer
 * Renders segmentation masks and probability maps with zoom/pan support
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { Layer } from './Layer';
import { LayerType, LayerConfig, LayerRenderOptions, SegmentationData, Transform } from './types';
import { VideoLayer } from './VideoLayer';

export class SegmentationLayer extends Layer {
  private currentData: SegmentationData | null = null;

  constructor(config?: Partial<LayerConfig>) {
    super({
      type: LayerType.SEGMENTATION,
      zIndex: 1,
      visible: true,
      opacity: 0.7, // Increased for visibility
      blendMode: 'source-over', // Normal blend for better visibility
      ...config,
    });
  }

  /**
   * Update segmentation data
   */
  updateData(data: SegmentationData): void {
    this.currentData = data;
    this.markDirty();
  }

  /**
   * Render segmentation overlay
   */
  render(options: LayerRenderOptions): void {
    if (!this.config.visible || !this.currentData) {
      return;
    }

    const ctx = this.context;

    // Clear canvas
    this.clear();

    // Save context state
    ctx.save();

    // Apply quality-based opacity
    const opacity = this.getQualityOpacity(options.quality);
    ctx.globalAlpha = opacity;

    // Set blend mode if specified
    if (this.config.blendMode) {
      ctx.globalCompositeOperation = this.config.blendMode;
    }

    // Render probability map if available
    if (this.currentData.probabilityMap) {
      this.renderProbabilityMap(ctx, this.currentData.probabilityMap, options);
    }

    // Render binary mask if available
    if (this.currentData.mask) {
      this.renderMask(ctx, this.currentData.mask, this.currentData.color, options);
    }

    // Restore context state
    ctx.restore();

    // Clear dirty flag
    this.clearDirty();
  }

  /**
   * Render probability map with color gradient
   */
  private renderProbabilityMap(
    ctx: CanvasRenderingContext2D,
    probMap: ImageData,
    options: LayerRenderOptions
  ): void {
    // Create colored version of probability map
    const colored = new ImageData(probMap.width, probMap.height);

    for (let i = 0; i < probMap.data.length; i += 4) {
      const prob = probMap.data[i]; // Grayscale value (0-255)

      // Color gradient: blue (low) → yellow (medium) → red (high)
      if (prob < 128) {
        // Blue to yellow
        const t = prob / 128;
        colored.data[i] = Math.round(255 * t); // R
        colored.data[i + 1] = Math.round(255 * t); // G
        colored.data[i + 2] = 255; // B
      } else {
        // Yellow to red
        const t = (prob - 128) / 127;
        colored.data[i] = 255; // R
        colored.data[i + 1] = Math.round(255 * (1 - t)); // G
        colored.data[i + 2] = 0; // B
      }

      // Increased opacity for better visibility (1.8x multiplier, max 255)
      const enhancedAlpha = Math.min(255, Math.round(prob * 1.8));
      colored.data[i + 3] = enhancedAlpha;
    }

    // Draw to canvas with zoom/pan transform from VideoLayer
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = probMap.width;
    tempCanvas.height = probMap.height;
    const tempCtx = tempCanvas.getContext('2d');
    if (tempCtx) {
      tempCtx.putImageData(colored, 0, 0);

      // Get zoom/pan transform from VideoLayer (via options.layerManager)
      const transform = this.getVideoLayerTransform(options);
      if (transform) {
        const { scale, offsetX, offsetY } = transform;
        const scaledWidth = probMap.width * scale;
        const scaledHeight = probMap.height * scale;
        ctx.drawImage(tempCanvas, offsetX, offsetY, scaledWidth, scaledHeight);
      } else {
        // Fallback: fit to canvas (legacy behavior)
        ctx.drawImage(tempCanvas, 0, 0, this.canvas.width, this.canvas.height);
      }
    }
  }

  /**
   * Render binary mask with solid color
   */
  private renderMask(
    ctx: CanvasRenderingContext2D,
    mask: ImageData,
    color: string | undefined,
    options: LayerRenderOptions
  ): void {
    const colored = new ImageData(mask.width, mask.height);
    const rgb = this.hexToRgb(color || '#FF0000'); // Default red for vessel

    // CRITICAL: Threshold for binary mask (fixes interpolation artifacts from PNG decode)
    // PNG decoding may introduce intermediate values (1-254) due to smoothing
    // We need to threshold back to binary (0 or 255)
    const THRESHOLD = 127; // Midpoint threshold

    for (let i = 0; i < mask.data.length; i += 4) {
      const value = mask.data[i]; // Grayscale value

      // Only render if value is above threshold (binary mask restoration)
      if (value > THRESHOLD) {
        colored.data[i] = rgb.r;
        colored.data[i + 1] = rgb.g;
        colored.data[i + 2] = rgb.b;
        colored.data[i + 3] = 255; // Full opacity for mask pixels
      }
    }

    // Draw to canvas with zoom/pan transform from VideoLayer
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = mask.width;
    tempCanvas.height = mask.height;
    const tempCtx = tempCanvas.getContext('2d');
    if (tempCtx) {
      tempCtx.putImageData(colored, 0, 0);

      // Get zoom/pan transform from VideoLayer (via options.layerManager)
      const transform = this.getVideoLayerTransform(options);
      if (transform) {
        const { scale, offsetX, offsetY } = transform;
        const scaledWidth = mask.width * scale;
        const scaledHeight = mask.height * scale;
        ctx.drawImage(tempCanvas, offsetX, offsetY, scaledWidth, scaledHeight);
      } else {
        // Fallback: fit to canvas (legacy behavior)
        ctx.drawImage(tempCanvas, 0, 0, this.canvas.width, this.canvas.height);
      }
    }
  }

  /**
   * Get zoom/pan transform from VideoLayer
   */
  private getVideoLayerTransform(options: LayerRenderOptions): Transform | null {
    // Check if layerManager is available in options
    if (!options.layerManager) {
      return null;
    }

    // Get VideoLayer from LayerManager
    const videoLayer = options.layerManager.getLayer(LayerType.VIDEO) as VideoLayer | undefined;
    if (!videoLayer || typeof videoLayer.getTransform !== 'function') {
      return null;
    }

    // Get transform from VideoLayer
    return videoLayer.getTransform();
  }

  /**
   * Convert hex color to RGB
   */
  private hexToRgb(hex: string): { r: number; g: number; b: number } {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : { r: 255, g: 0, b: 0 }; // Default red
  }

  /**
   * Clear segmentation data
   */
  clearData(): void {
    this.currentData = null;
    this.clear();
  }

  /**
   * Get current segmentation data
   */
  getData(): SegmentationData | null {
    return this.currentData ? { ...this.currentData } : null;
  }
}
