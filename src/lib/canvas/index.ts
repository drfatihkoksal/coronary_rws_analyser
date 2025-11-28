/**
 * Canvas Layer System
 * Multi-layer rendering system for medical imaging
 *
 * Exports all layer classes, types, and utilities
 */

// Types
export * from './types';

// Base class
export { Layer } from './Layer';

// Layer implementations
export { VideoLayer } from './VideoLayer';
export { SegmentationLayer } from './SegmentationLayer';
export { AnnotationLayer } from './AnnotationLayer';
export { OverlayLayer } from './OverlayLayer';

// Manager
export { LayerManager } from './LayerManager';
