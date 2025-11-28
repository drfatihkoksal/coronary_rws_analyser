/**
 * Overlay Layer
 * Renders ECG overlay and UI elements on top of video
 *
 * Features:
 * - Transparent ECG waveform overlay at bottom
 * - R-peak markers with vertical lines
 * - Progress bar with scrubber handle
 * - Frame counter and heart rate display
 * - Timeline click-to-seek support
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { Layer } from './Layer';
import {
  LayerType,
  LayerConfig,
  LayerRenderOptions,
  OverlayData,
  EKGData,
  UIElement,
} from './types';

export class OverlayLayer extends Layer {
  private currentData: OverlayData = {};

  // ECG area dimensions (for mouse interaction)
  private ekgHeight = 80;
  private ekgPadding = 10;

  constructor(config?: Partial<LayerConfig>) {
    super({
      type: LayerType.OVERLAY,
      zIndex: 3,
      visible: true,
      opacity: 0.9,
      ...config,
    });
  }

  /**
   * Update overlay data
   */
  updateData(data: Partial<OverlayData>): void {
    this.currentData = { ...this.currentData, ...data };
    this.markDirty();
  }

  /**
   * Render overlay elements
   */
  render(_options: LayerRenderOptions): void {
    if (!this.config.visible) {
      return;
    }

    const ctx = this.context;

    // Clear canvas
    this.clear();

    // Save context state
    ctx.save();

    // Apply opacity
    ctx.globalAlpha = this.config.opacity;

    // Render EKG if available
    if (this.currentData.ekg) {
      this.renderEKG(ctx, this.currentData.ekg);
    }

    // Render UI elements
    if (this.currentData.uiElements) {
      this.currentData.uiElements.forEach((element) => {
        this.renderUIElement(ctx, element);
      });
    }

    // Restore context state
    ctx.restore();

    // Clear dirty flag
    this.clearDirty();
  }

  /**
   * Get EKG area bounds (for mouse interaction/timeline scrubbing)
   */
  getEKGBounds(): { x: number; y: number; width: number; height: number } | null {
    if (!this.currentData.ekg || this.currentData.ekg.signal.length === 0) {
      return null;
    }

    const y = this.canvas.height - this.ekgHeight - this.ekgPadding;

    return {
      x: this.ekgPadding,
      y: y,
      width: this.canvas.width - 2 * this.ekgPadding,
      height: this.ekgHeight,
    };
  }

  /**
   * Check if a point is in the EKG timeline area
   */
  isInEKGArea(canvasX: number, canvasY: number): boolean {
    const bounds = this.getEKGBounds();
    if (!bounds) return false;

    return (
      canvasX >= bounds.x &&
      canvasX <= bounds.x + bounds.width &&
      canvasY >= bounds.y &&
      canvasY <= bounds.y + bounds.height
    );
  }

  /**
   * Get frame from EKG timeline click position
   */
  getFrameFromClick(canvasX: number): number {
    const bounds = this.getEKGBounds();
    if (!bounds || !this.currentData.ekg?.totalFrames) {
      return 0;
    }

    // Calculate relative position in EKG area
    const relativeX = canvasX - bounds.x;
    const percentage = Math.max(0, Math.min(1, relativeX / bounds.width));
    const frame = Math.floor(percentage * (this.currentData.ekg.totalFrames - 1));

    return Math.max(0, Math.min(this.currentData.ekg.totalFrames - 1, frame));
  }

  /**
   * Render EKG overlay at bottom of canvas
   */
  private renderEKG(ctx: CanvasRenderingContext2D, ekg: EKGData): void {
    if (ekg.signal.length === 0) return;

    const height = this.ekgHeight;
    const padding = this.ekgPadding;
    const y = this.canvas.height - height - padding;
    const width = this.canvas.width - 2 * padding;

    // Semi-transparent background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(padding, y, width, height);

    // Calculate scaling
    const minSignal = Math.min(...ekg.signal);
    const maxSignal = Math.max(...ekg.signal);
    const signalRange = maxSignal - minSignal || 1;

    const xScale = width / (ekg.signal.length - 1);
    const yScale = (height - 20) / signalRange;

    // Draw EKG waveform
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < ekg.signal.length; i++) {
      const x = padding + i * xScale;
      const signalY = y + height - 10 - (ekg.signal[i] - minSignal) * yScale;

      if (i === 0) {
        ctx.moveTo(x, signalY);
      } else {
        ctx.lineTo(x, signalY);
      }
    }

    ctx.stroke();

    // Draw R-peaks
    if (ekg.rPeaks && ekg.rPeaks.length > 0) {
      ctx.fillStyle = '#FF0000';

      ekg.rPeaks.forEach((peakIdx) => {
        if (peakIdx < ekg.signal.length) {
          const x = padding + peakIdx * xScale;
          const signalY = y + height - 10 - (ekg.signal[peakIdx] - minSignal) * yScale;

          // R-peak circle
          ctx.beginPath();
          ctx.arc(x, signalY, 3, 0, 2 * Math.PI);
          ctx.fill();

          // Vertical line through R-peak
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, y + 10);
          ctx.lineTo(x, y + height - 10);
          ctx.stroke();
        }
      });
    }

    // Draw progress bar and scrubber
    if (
      ekg.currentFrame !== undefined &&
      ekg.totalFrames !== undefined &&
      ekg.totalFrames > 0
    ) {
      const frameRatio = ekg.currentFrame / Math.max(1, ekg.totalFrames - 1);
      const progressWidth = frameRatio * width;

      // Progress fill
      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'; // primary/30
      ctx.fillRect(padding, y, progressWidth, height);

      // Current frame marker line
      const markerX = padding + progressWidth;
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(markerX, y);
      ctx.lineTo(markerX, y + height);
      ctx.stroke();

      // Scrubber handle
      ctx.fillStyle = '#3B82F6'; // primary
      ctx.strokeStyle = '#1E3A8A'; // darker border
      ctx.lineWidth = 1;

      const scrubberWidth = 12;
      const scrubberHeight = 24;
      const scrubberX = markerX - scrubberWidth / 2;
      const scrubberY = y + height / 2 - scrubberHeight / 2;

      // Rounded rectangle for scrubber
      this.drawRoundedRect(
        ctx,
        scrubberX,
        scrubberY,
        scrubberWidth,
        scrubberHeight,
        3
      );
      ctx.fill();
      ctx.stroke();

      // Frame count text (bottom left)
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(
        `${ekg.currentFrame + 1} / ${ekg.totalFrames}`,
        padding + 5,
        y + 5
      );
    }

    // Draw heart rate (top right of EKG area)
    if (ekg.heartRate) {
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(
        `HR: ${Math.round(ekg.heartRate)} bpm`,
        this.canvas.width - padding - 5,
        y + 5
      );
    }

    // Draw hint text
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '9px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(
      'Click to seek • Drag to scrub',
      this.canvas.width / 2,
      y + height - 3
    );
  }

  /**
   * Draw rounded rectangle helper
   */
  private drawRoundedRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  /**
   * Render UI element
   */
  private renderUIElement(ctx: CanvasRenderingContext2D, element: UIElement): void {
    ctx.fillStyle = element.color || '#FFFFFF';
    ctx.strokeStyle = element.color || '#FFFFFF';

    switch (element.type) {
      case 'text':
        ctx.font = `${element.fontSize || 12}px Arial`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        if (element.text) {
          ctx.fillText(element.text, element.x, element.y);
        }
        break;

      case 'line':
        if (element.width !== undefined && element.height !== undefined) {
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(element.x, element.y);
          ctx.lineTo(element.x + element.width, element.y + element.height);
          ctx.stroke();
        }
        break;

      case 'rect':
        if (element.width !== undefined && element.height !== undefined) {
          ctx.strokeRect(element.x, element.y, element.width, element.height);
        }
        break;

      case 'circle':
        if (element.radius !== undefined) {
          ctx.beginPath();
          ctx.arc(element.x, element.y, element.radius, 0, 2 * Math.PI);
          ctx.fill();
        }
        break;
    }
  }

  /**
   * Clear all overlay data
   */
  clearData(): void {
    this.currentData = {};
    this.clear();
  }

  /**
   * Get current overlay data
   */
  getData(): OverlayData {
    return { ...this.currentData };
  }

  /**
   * Update only EKG data (convenience method)
   */
  updateEKG(ekg: Partial<EKGData>): void {
    if (this.currentData.ekg) {
      this.currentData.ekg = { ...this.currentData.ekg, ...ekg };
    } else {
      this.currentData.ekg = ekg as EKGData;
    }
    this.markDirty();
  }

  /**
   * Update current frame (for scrubber position)
   */
  setCurrentFrame(frame: number): void {
    if (this.currentData.ekg) {
      this.currentData.ekg.currentFrame = frame;
      this.markDirty();
    }
  }
}
