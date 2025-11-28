/**
 * Canvas Layer System - Type Definitions
 * Multi-layer rendering system for medical imaging
 *
 * Ported from Coronary Clear Vision V2.2
 */

export enum LayerType {
  VIDEO = 'video',
  SEGMENTATION = 'segmentation',
  ANNOTATION = 'annotation',
  OVERLAY = 'overlay',
}

export enum RenderingMode {
  WEBGL2 = 'webgl2',
  WEBGL = 'webgl',
  CANVAS2D = 'canvas2d',
}

export enum QualityLevel {
  HIGH = 'high', // Paused state: 0.7 opacity
  LOW = 'low', // Playing state: 0.3 opacity
}

export interface LayerConfig {
  type: LayerType;
  zIndex: number;
  visible: boolean;
  opacity: number;
  blendMode?: GlobalCompositeOperation;
}

export interface RenderContext {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D | WebGLRenderingContext | WebGL2RenderingContext;
  mode: RenderingMode;
}

export interface LayerRenderOptions {
  quality: QualityLevel;
  dirtyRegion?: DOMRect;
  timestamp?: number;
  layerManager?: LayerManagerInterface; // For accessing other layers (e.g., VideoLayer transform)
}

export interface PerformanceMetrics {
  fps: number;
  renderTime: number;
  layerCount: number;
  mode: RenderingMode;
}

// Forward declaration for LayerManager interface
export interface LayerManagerInterface {
  getLayer(type: LayerType): unknown;
  render(timestamp?: number): void;
  resize(width: number, height: number): void;
}

// Data interfaces for each layer
export interface VideoLayerData {
  frameData: string; // base64 image data
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface ROI {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Centerline {
  points: Point[];
  color?: string;
  width?: number;
}

export interface DiameterMarker {
  point: Point;
  diameter: number;
  type: 'MLD' | 'Proximal' | 'Distal';
  color?: string;
}

export interface YoloKeypoint {
  point: Point;
  name: string;  // 'start', 'quarter', 'center', '3/4', 'end'
  confidence?: number;
}

export interface AnnotationData {
  seedPoints?: Point[];
  yoloKeypoints?: YoloKeypoint[];  // YOLO detected keypoints
  roi?: ROI;
  centerlines?: Centerline[];
  diameterMarkers?: DiameterMarker[];
}

export interface SegmentationData {
  mask?: ImageData; // Binary segmentation mask
  probabilityMap?: ImageData; // Probability map (0-255 values)
  color?: string; // Mask overlay color
}

export interface EKGData {
  signal: number[]; // Signal values
  time: number[]; // Time points
  rPeaks: number[]; // R-peak indices (in signal samples)
  currentFrame?: number; // Current frame index for position marker
  totalFrames?: number; // Total number of video frames
  samplingRate?: number;
  heartRate?: number;
  duration?: number; // Total ECG duration in seconds
}

export interface UIElement {
  type: 'text' | 'line' | 'rect' | 'circle';
  x: number;
  y: number;
  width?: number;
  height?: number;
  radius?: number;
  text?: string;
  color?: string;
  fontSize?: number;
}

export interface OverlayData {
  ekg?: EKGData;
  uiElements?: UIElement[];
}

// Transform interface for coordinate conversion
export interface Transform {
  scale: number;
  offsetX: number;
  offsetY: number;
}
