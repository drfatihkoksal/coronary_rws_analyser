/**
 * Video Layer
 * Renders base DICOM frame data with zoom and pan support
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { Layer } from './Layer';
import { LayerType, LayerConfig, LayerRenderOptions, VideoLayerData, Transform } from './types';

export class VideoLayer extends Layer {
  private currentImage: HTMLImageElement | null = null;
  private imageLoaded: boolean = false;

  // Zoom and Pan state
  private zoomLevel: number = 1.0; // 1.0 = fit to canvas, 2.0 = 200%, 0.5 = 50%
  private panOffsetX: number = 0; // Pan offset in canvas pixels
  private panOffsetY: number = 0;
  private minZoom: number = 0.1; // Minimum 10%
  private maxZoom: number = 10.0; // Maximum 1000%

  constructor(config?: Partial<LayerConfig>) {
    super({
      type: LayerType.VIDEO,
      zIndex: 0,
      visible: true,
      opacity: 1.0,
      ...config,
    });
  }

  /**
   * Load and cache frame image
   */
  async loadFrame(data: VideoLayerData): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.onload = () => {
        this.currentImage = img;
        this.imageLoaded = true;
        this.markDirty();
        resolve();
      };

      img.onerror = () => {
        reject(new Error('Failed to load frame image'));
      };

      // Handle both with and without data URL prefix
      if (data.frameData.startsWith('data:')) {
        img.src = data.frameData;
      } else {
        img.src = `data:image/png;base64,${data.frameData}`;
      }
    });
  }

  /**
   * Render video frame to canvas with zoom and pan
   */
  render(_options: LayerRenderOptions): void {
    if (!this.config.visible || !this.imageLoaded || !this.currentImage) {
      return;
    }

    const ctx = this.context;
    const img = this.currentImage;

    // Clear canvas
    this.clear();

    // Save context state
    ctx.save();

    // Apply opacity
    ctx.globalAlpha = this.config.opacity;

    // Calculate base scaling to fit canvas
    const baseScale = Math.min(
      this.canvas.width / img.width,
      this.canvas.height / img.height
    );

    // Apply zoom level
    const scale = baseScale * this.zoomLevel;

    // Calculate base center position (without zoom)
    const baseCenterX = this.canvas.width / 2;
    const baseCenterY = this.canvas.height / 2;

    // Calculate zoomed image dimensions
    const zoomedWidth = img.width * scale;
    const zoomedHeight = img.height * scale;

    // Apply pan offset (centered + pan)
    const x = baseCenterX - zoomedWidth / 2 + this.panOffsetX;
    const y = baseCenterY - zoomedHeight / 2 + this.panOffsetY;

    // Draw image with zoom and pan
    ctx.drawImage(img, x, y, zoomedWidth, zoomedHeight);

    // Restore context state
    ctx.restore();

    // Clear dirty flag
    this.clearDirty();
  }

  /**
   * Get transform information for coordinate mapping (with zoom and pan)
   */
  getTransform(): Transform {
    if (!this.currentImage) {
      return { scale: 1, offsetX: 0, offsetY: 0 };
    }

    // Calculate base scaling to fit canvas
    const baseScale = Math.min(
      this.canvas.width / this.currentImage.width,
      this.canvas.height / this.currentImage.height
    );

    // Apply zoom level
    const scale = baseScale * this.zoomLevel;

    // Calculate base center position
    const baseCenterX = this.canvas.width / 2;
    const baseCenterY = this.canvas.height / 2;

    // Calculate zoomed image dimensions
    const zoomedWidth = this.currentImage.width * scale;
    const zoomedHeight = this.currentImage.height * scale;

    // Apply pan offset
    const offsetX = baseCenterX - zoomedWidth / 2 + this.panOffsetX;
    const offsetY = baseCenterY - zoomedHeight / 2 + this.panOffsetY;

    return { scale, offsetX, offsetY };
  }

  /**
   * Convert canvas coordinates to image coordinates (with zoom and pan)
   */
  canvasToImage(canvasX: number, canvasY: number): { x: number; y: number } {
    if (!this.currentImage) {
      return { x: canvasX, y: canvasY };
    }

    const { scale, offsetX, offsetY } = this.getTransform();

    const x = (canvasX - offsetX) / scale;
    const y = (canvasY - offsetY) / scale;

    return { x, y };
  }

  /**
   * Convert image coordinates to canvas coordinates (with zoom and pan)
   */
  imageToCanvas(imageX: number, imageY: number): { x: number; y: number } {
    if (!this.currentImage) {
      return { x: imageX, y: imageY };
    }

    const { scale, offsetX, offsetY } = this.getTransform();

    const x = imageX * scale + offsetX;
    const y = imageY * scale + offsetY;

    return { x, y };
  }

  /**
   * Set zoom level
   * @param level - Zoom level (1.0 = 100%, 2.0 = 200%, etc.)
   * @param centerX - Optional canvas X coordinate to zoom towards (default: canvas center)
   * @param centerY - Optional canvas Y coordinate to zoom towards (default: canvas center)
   */
  setZoom(level: number, centerX?: number, centerY?: number): void {
    const oldZoom = this.zoomLevel;
    const newZoom = Math.max(this.minZoom, Math.min(this.maxZoom, level));

    if (oldZoom === newZoom) return;

    // If zooming towards a specific point, adjust pan offset to keep that point centered
    if (centerX !== undefined && centerY !== undefined && this.currentImage) {
      const { scale: oldScale, offsetX: oldOffsetX, offsetY: oldOffsetY } = this.getTransform();

      // Calculate image point that was under the cursor
      const imageX = (centerX - oldOffsetX) / oldScale;
      const imageY = (centerY - oldOffsetY) / oldScale;

      // Update zoom
      this.zoomLevel = newZoom;

      // Calculate new scale
      const { scale: newScale } = this.getTransform();

      // Calculate where that image point should be now
      const newCanvasX = imageX * newScale;
      const newCanvasY = imageY * newScale;

      // Adjust pan offset to keep the point under cursor
      const baseCenterX = this.canvas.width / 2;
      const baseCenterY = this.canvas.height / 2;
      const zoomedWidth = this.currentImage.width * newScale;
      const zoomedHeight = this.currentImage.height * newScale;

      const baseOffsetX = baseCenterX - zoomedWidth / 2;
      const baseOffsetY = baseCenterY - zoomedHeight / 2;

      this.panOffsetX = centerX - newCanvasX - baseOffsetX;
      this.panOffsetY = centerY - newCanvasY - baseOffsetY;
    } else {
      // Simple zoom without preserving point
      this.zoomLevel = newZoom;
    }

    this.markDirty();
  }

  /**
   * Zoom in by a factor (default 1.2x)
   */
  zoomIn(factor: number = 1.2, centerX?: number, centerY?: number): void {
    this.setZoom(this.zoomLevel * factor, centerX, centerY);
  }

  /**
   * Zoom out by a factor (default 1.2x)
   */
  zoomOut(factor: number = 1.2, centerX?: number, centerY?: number): void {
    this.setZoom(this.zoomLevel / factor, centerX, centerY);
  }

  /**
   * Set pan offset in canvas pixels
   */
  setPan(offsetX: number, offsetY: number): void {
    this.panOffsetX = offsetX;
    this.panOffsetY = offsetY;
    this.markDirty();
  }

  /**
   * Adjust pan offset by delta
   */
  pan(deltaX: number, deltaY: number): void {
    this.panOffsetX += deltaX;
    this.panOffsetY += deltaY;
    this.markDirty();
  }

  /**
   * Reset zoom and pan to default (fit to canvas, centered)
   */
  resetView(): void {
    this.zoomLevel = 1.0;
    this.panOffsetX = 0;
    this.panOffsetY = 0;
    this.markDirty();
  }

  /**
   * Get current zoom level
   */
  getZoomLevel(): number {
    return this.zoomLevel;
  }

  /**
   * Get current pan offset
   */
  getPanOffset(): { x: number; y: number } {
    return { x: this.panOffsetX, y: this.panOffsetY };
  }

  /**
   * Get image dimensions
   */
  getImageDimensions(): { width: number; height: number } | null {
    if (!this.currentImage) return null;
    return { width: this.currentImage.width, height: this.currentImage.height };
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.currentImage = null;
    this.imageLoaded = false;
    super.dispose();
  }
}
