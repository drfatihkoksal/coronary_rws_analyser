/**
 * Base Layer Class
 * Abstract base for all canvas layers
 *
 * Ported from Coronary Clear Vision V2.2
 */

import {
  LayerType,
  LayerConfig,
  LayerRenderOptions,
  QualityLevel,
} from './types';

export abstract class Layer {
  protected config: LayerConfig;
  protected canvas: HTMLCanvasElement;
  protected context: CanvasRenderingContext2D;
  protected dirty: boolean = true;

  constructor(config: LayerConfig) {
    this.config = config;
    this.canvas = document.createElement('canvas');
    const ctx = this.canvas.getContext('2d', { alpha: true });
    if (!ctx) throw new Error('Failed to create canvas context');
    this.context = ctx;
  }

  /**
   * Abstract render method - must be implemented by subclasses
   */
  abstract render(options: LayerRenderOptions): void;

  /**
   * Update layer configuration
   */
  updateConfig(partial: Partial<LayerConfig>): void {
    this.config = { ...this.config, ...partial };
    this.markDirty();
  }

  /**
   * Set layer visibility
   */
  setVisible(visible: boolean): void {
    this.config.visible = visible;
    this.markDirty();
  }

  /**
   * Set layer opacity
   */
  setOpacity(opacity: number): void {
    this.config.opacity = Math.max(0, Math.min(1, opacity));
    this.markDirty();
  }

  /**
   * Set layer z-index
   */
  setZIndex(zIndex: number): void {
    this.config.zIndex = zIndex;
  }

  /**
   * Mark layer as dirty (needs re-render)
   */
  markDirty(): void {
    this.dirty = true;
  }

  /**
   * Check if layer needs re-render
   */
  isDirty(): boolean {
    return this.dirty;
  }

  /**
   * Clear dirty flag
   */
  clearDirty(): void {
    this.dirty = false;
  }

  /**
   * Resize canvas
   */
  resize(width: number, height: number): void {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.markDirty();
    }
  }

  /**
   * Clear canvas
   */
  clear(): void {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Get canvas element
   */
  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  /**
   * Get layer configuration
   */
  getConfig(): LayerConfig {
    return { ...this.config };
  }

  /**
   * Get layer type
   */
  getType(): LayerType {
    return this.config.type;
  }

  /**
   * Check if layer is visible
   */
  isVisible(): boolean {
    return this.config.visible;
  }

  /**
   * Get opacity for current quality level
   */
  protected getQualityOpacity(quality: QualityLevel): number {
    const baseOpacity = this.config.opacity;

    // For video layer, always use base opacity
    if (this.config.type === LayerType.VIDEO) {
      return baseOpacity;
    }

    // For other layers, adjust based on quality
    switch (quality) {
      case QualityLevel.HIGH:
        return baseOpacity * 0.7;
      case QualityLevel.LOW:
        return baseOpacity * 0.3;
      default:
        return baseOpacity;
    }
  }

  /**
   * Dispose layer resources
   */
  dispose(): void {
    // Clean up resources
    this.canvas.remove();
  }
}
