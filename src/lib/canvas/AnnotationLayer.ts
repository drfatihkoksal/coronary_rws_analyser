/**
 * Annotation Layer
 * Renders seed points, ROI boxes, centerlines, and QCA measurements
 *
 * Ported from Coronary Clear Vision V2.2
 */

import { Layer } from './Layer';
import { VideoLayer } from './VideoLayer';
import {
  LayerType,
  LayerConfig,
  LayerRenderOptions,
  AnnotationData,
  Point,
  ROI,
  Centerline,
  DiameterMarker,
  YoloKeypoint,
  Transform,
} from './types';

interface VisibilityFlags {
  roi: boolean;
  centerline: boolean;
  seedPoints: boolean;
  yoloKeypoints: boolean;
  diameterMarkers: boolean;
}

export class AnnotationLayer extends Layer {
  private currentData: AnnotationData = {};
  private visibilityFlags: VisibilityFlags = {
    roi: true,
    centerline: true,
    seedPoints: true,
    yoloKeypoints: true,
    diameterMarkers: true,
  };

  constructor(config?: Partial<LayerConfig>) {
    super({
      type: LayerType.ANNOTATION,
      zIndex: 2,
      visible: true,
      opacity: 1.0,
      ...config,
    });
  }

  /**
   * Update annotation data
   */
  updateData(data: Partial<AnnotationData>): void {
    this.currentData = { ...this.currentData, ...data };
    this.markDirty();
  }

  /**
   * Add seed point
   */
  addSeedPoint(point: Point): void {
    if (!this.currentData.seedPoints) {
      this.currentData.seedPoints = [];
    }
    this.currentData.seedPoints.push(point);
    this.markDirty();
  }

  /**
   * Remove all seed points
   */
  clearSeedPoints(): void {
    this.currentData.seedPoints = [];
    this.markDirty();
  }

  /**
   * Set ROI
   */
  setROI(roi: ROI | undefined): void {
    this.currentData.roi = roi;
    this.markDirty();
  }

  /**
   * Set YOLO keypoints
   */
  setYoloKeypoints(keypoints: YoloKeypoint[]): void {
    this.currentData.yoloKeypoints = keypoints;
    this.markDirty();
  }

  /**
   * Set visibility flags
   */
  setVisibility(flags: Partial<VisibilityFlags>): void {
    this.visibilityFlags = { ...this.visibilityFlags, ...flags };
    this.markDirty();
  }

  /**
   * Get visibility flags
   */
  getVisibility(): VisibilityFlags {
    return { ...this.visibilityFlags };
  }

  /**
   * Get zoom/pan transform from VideoLayer
   */
  private getVideoLayerTransform(options: LayerRenderOptions): Transform | null {
    if (!options.layerManager) {
      return null;
    }

    const videoLayer = options.layerManager.getLayer(LayerType.VIDEO) as VideoLayer | undefined;
    if (!videoLayer || typeof videoLayer.getTransform !== 'function') {
      return null;
    }

    return videoLayer.getTransform();
  }

  /**
   * Transform image coordinates to canvas coordinates
   */
  private transformPoint(point: Point, transform: Transform | null): Point {
    if (!transform) {
      return point;
    }
    return {
      x: point.x * transform.scale + transform.offsetX,
      y: point.y * transform.scale + transform.offsetY,
    };
  }

  /**
   * Render annotations
   */
  render(options: LayerRenderOptions): void {
    if (!this.config.visible) {
      return;
    }

    const ctx = this.context;

    // Clear canvas
    this.clear();

    // Save context state
    ctx.save();

    // Get transform from VideoLayer for zoom/pan sync
    const transform = this.getVideoLayerTransform(options);

    // Apply opacity (annotations always visible, no quality-based opacity)
    ctx.globalAlpha = this.config.opacity;

    // Render in order: ROI → Centerlines → Diameter markers → YOLO keypoints → Seed points
    if (this.currentData.roi && this.visibilityFlags.roi) {
      this.renderROI(ctx, this.currentData.roi, transform);
    }

    if (this.currentData.centerlines && this.visibilityFlags.centerline) {
      this.currentData.centerlines.forEach((centerline) => {
        this.renderCenterline(ctx, centerline, transform);
      });
    }

    if (this.currentData.diameterMarkers && this.visibilityFlags.diameterMarkers) {
      this.currentData.diameterMarkers.forEach((marker) => {
        this.renderDiameterMarker(ctx, marker, transform);
      });
    }

    if (this.currentData.yoloKeypoints && this.visibilityFlags.yoloKeypoints) {
      this.currentData.yoloKeypoints.forEach((keypoint) => {
        this.renderYoloKeypoint(ctx, keypoint, transform);
      });
    }

    if (this.currentData.seedPoints && this.visibilityFlags.seedPoints) {
      this.currentData.seedPoints.forEach((point, index) => {
        this.renderSeedPoint(ctx, point, index, transform);
      });
    }

    // Restore context state
    ctx.restore();

    // Clear dirty flag
    this.clearDirty();
  }

  /**
   * Render seed point as crosshair
   */
  private renderSeedPoint(ctx: CanvasRenderingContext2D, point: Point, index: number, transform: Transform | null): void {
    const tp = this.transformPoint(point, transform);
    const size = 12; // Crosshair size

    // Draw crosshair (white outer stroke for visibility)
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(tp.x - size, tp.y);
    ctx.lineTo(tp.x + size, tp.y);
    ctx.stroke();

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(tp.x, tp.y - size);
    ctx.lineTo(tp.x, tp.y + size);
    ctx.stroke();

    // Draw crosshair (red inner line)
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 2;

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(tp.x - size, tp.y);
    ctx.lineTo(tp.x + size, tp.y);
    ctx.stroke();

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(tp.x, tp.y - size);
    ctx.lineTo(tp.x, tp.y + size);
    ctx.stroke();

    // Draw index number with background
    const label = (index + 1).toString();
    ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Text background
    const metrics = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(
      tp.x - metrics.width / 2 - 2,
      tp.y + size + 2,
      metrics.width + 4,
      13
    );

    // Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, tp.x, tp.y + size + 3);
  }

  /**
   * Render YOLO keypoint (diamond shape, green color)
   */
  private renderYoloKeypoint(ctx: CanvasRenderingContext2D, keypoint: YoloKeypoint, transform: Transform | null): void {
    const { point, name } = keypoint;
    const tp = this.transformPoint(point, transform);
    const size = 8;

    // Draw diamond shape (rotated square)
    ctx.save();
    ctx.translate(tp.x, tp.y);
    ctx.rotate(Math.PI / 4);

    // White border
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(-size / 2 - 1, -size / 2 - 1, size + 2, size + 2);

    // Green fill
    ctx.fillStyle = '#00FF00';
    ctx.fillRect(-size / 2, -size / 2, size, size);

    ctx.restore();

    // Draw label with background
    ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const metrics = ctx.measureText(name);
    const labelY = tp.y + size + 4;

    // Text background
    ctx.fillStyle = 'rgba(0, 100, 0, 0.8)';
    ctx.fillRect(
      tp.x - metrics.width / 2 - 3,
      labelY - 1,
      metrics.width + 6,
      12
    );

    // Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(name, tp.x, labelY);
  }

  /**
   * Render ROI bounding box
   */
  private renderROI(ctx: CanvasRenderingContext2D, roi: ROI, transform: Transform | null): void {
    // Transform ROI corners
    const topLeft = this.transformPoint({ x: roi.x, y: roi.y }, transform);
    const bottomRight = this.transformPoint({ x: roi.x + roi.width, y: roi.y + roi.height }, transform);
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;

    // Draw dashed rectangle
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(topLeft.x, topLeft.y, width, height);
    ctx.setLineDash([]); // Reset dash

    // Draw corner handles
    const handleSize = 8;
    const corners = [
      { x: topLeft.x, y: topLeft.y },
      { x: bottomRight.x, y: topLeft.y },
      { x: topLeft.x, y: bottomRight.y },
      { x: bottomRight.x, y: bottomRight.y },
    ];

    ctx.fillStyle = '#00FF00';
    corners.forEach((corner) => {
      ctx.fillRect(
        corner.x - handleSize / 2,
        corner.y - handleSize / 2,
        handleSize,
        handleSize
      );
    });

    // Draw label
    ctx.fillStyle = '#00FF00';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('ROI', topLeft.x, topLeft.y - 5);
  }

  /**
   * Render centerline
   */
  private renderCenterline(ctx: CanvasRenderingContext2D, centerline: Centerline, transform: Transform | null): void {
    if (centerline.points.length < 2) return;

    ctx.strokeStyle = centerline.color || '#FFFF00';
    ctx.lineWidth = centerline.width || 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const firstPoint = this.transformPoint(centerline.points[0], transform);
    ctx.beginPath();
    ctx.moveTo(firstPoint.x, firstPoint.y);

    for (let i = 1; i < centerline.points.length; i++) {
      const tp = this.transformPoint(centerline.points[i], transform);
      ctx.lineTo(tp.x, tp.y);
    }

    ctx.stroke();
  }

  /**
   * Render diameter marker
   */
  private renderDiameterMarker(ctx: CanvasRenderingContext2D, marker: DiameterMarker, transform: Transform | null): void {
    const tp = this.transformPoint(marker.point, transform);

    // Color based on type
    const colors = {
      MLD: '#FF0000', // Red for MLD (most important)
      Proximal: '#00FFFF', // Cyan for proximal
      Distal: '#FFFF00', // Yellow for distal
    };

    const color = marker.color || colors[marker.type];
    const radius = 5;

    // Draw marker point
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(tp.x, tp.y, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Draw white border
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tp.x, tp.y, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw label with diameter value
    const label = `${marker.type}: ${marker.diameter.toFixed(2)}mm`;
    ctx.fillStyle = color;
    ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';

    // Draw text background
    const metrics = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(
      tp.x + 10,
      tp.y - 20,
      metrics.width + 4,
      14
    );

    // Draw text
    ctx.fillStyle = color;
    ctx.fillText(label, tp.x + 12, tp.y - 8);
  }

  /**
   * Clear all annotations
   */
  clearData(): void {
    this.currentData = {};
    this.clear();
  }

  /**
   * Get current annotation data
   */
  getData(): AnnotationData {
    return { ...this.currentData };
  }
}
