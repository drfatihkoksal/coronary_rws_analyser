/**
 * Player Store
 *
 * Manages video playback state including current frame, playback speed,
 * and view transform (zoom/pan).
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { PlaybackState, ViewTransform } from '../types';

export type AnnotationMode = 'none' | 'select' | 'roi' | 'fixed-roi' | 'seed' | 'pan';

interface PlayerState {
  // Playback
  currentFrame: number;
  playbackState: PlaybackState;
  playbackSpeed: number; // 0.25, 0.5, 1.0, 2.0
  isLooping: boolean;

  // Frame range
  totalFrames: number;
  frameRate: number;

  // View transform (shared across canvas layers)
  viewTransform: ViewTransform;
  viewTransformVersion: number; // Increment to trigger re-renders

  // Annotation mode (shared between Toolbar and AnnotationCanvas)
  annotationMode: AnnotationMode;

  // Actions
  setCurrentFrame: (frame: number) => void;
  nextFrame: () => void;
  previousFrame: () => void;
  play: () => void;
  pause: () => void;
  stop: () => void;
  togglePlayback: () => void;
  setPlaybackSpeed: (speed: number) => void;
  setLooping: (loop: boolean) => void;

  // Frame range
  setTotalFrames: (total: number) => void;
  setFrameRate: (rate: number) => void;

  // View transform
  setViewTransform: (transform: ViewTransform) => void;
  zoom: (delta: number, centerX?: number, centerY?: number) => void;
  pan: (deltaX: number, deltaY: number) => void;
  resetView: () => void;

  // Annotation mode
  setAnnotationMode: (mode: AnnotationMode) => void;
  toggleAnnotationMode: (mode: AnnotationMode) => void;

  // Reset
  reset: () => void;
}

const DEFAULT_VIEW_TRANSFORM: ViewTransform = {
  scale: 1,
  x: 0,
  y: 0,
};

export const usePlayerStore = create<PlayerState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    currentFrame: 0,
    playbackState: 'stopped',
    playbackSpeed: 1.0,
    isLooping: true,
    totalFrames: 0,
    frameRate: 15,
    viewTransform: { ...DEFAULT_VIEW_TRANSFORM },
    viewTransformVersion: 0,
    annotationMode: 'none',

    // Frame navigation
    setCurrentFrame: (frame) => {
      const { totalFrames } = get();
      if (totalFrames > 0) {
        // Clamp to valid range
        const clampedFrame = Math.max(0, Math.min(frame, totalFrames - 1));
        set({ currentFrame: clampedFrame });
      }
    },

    nextFrame: () => {
      const { currentFrame, totalFrames, isLooping } = get();
      if (totalFrames === 0) return;

      let nextFrame = currentFrame + 1;
      if (nextFrame >= totalFrames) {
        nextFrame = isLooping ? 0 : totalFrames - 1;
      }
      set({ currentFrame: nextFrame });
    },

    previousFrame: () => {
      const { currentFrame, totalFrames, isLooping } = get();
      if (totalFrames === 0) return;

      let prevFrame = currentFrame - 1;
      if (prevFrame < 0) {
        prevFrame = isLooping ? totalFrames - 1 : 0;
      }
      set({ currentFrame: prevFrame });
    },

    // Playback control
    play: () => set({ playbackState: 'playing' }),
    pause: () => set({ playbackState: 'paused' }),
    stop: () => set({ playbackState: 'stopped', currentFrame: 0 }),

    togglePlayback: () => {
      const { playbackState } = get();
      if (playbackState === 'playing') {
        set({ playbackState: 'paused' });
      } else {
        set({ playbackState: 'playing' });
      }
    },

    setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
    setLooping: (loop) => set({ isLooping: loop }),

    // Frame range
    setTotalFrames: (total) => set({ totalFrames: total }),
    setFrameRate: (rate) => set({ frameRate: rate }),

    // View transform
    setViewTransform: (transform) => {
      set((state) => ({
        viewTransform: transform,
        viewTransformVersion: state.viewTransformVersion + 1,
      }));
    },

    zoom: (delta, centerX, centerY) => {
      const { viewTransform } = get();
      const minScale = 0.1;
      const maxScale = 10;

      let newScale = viewTransform.scale * (1 + delta);
      newScale = Math.max(minScale, Math.min(maxScale, newScale));

      // Zoom toward cursor position if provided
      let newX = viewTransform.x;
      let newY = viewTransform.y;

      if (centerX !== undefined && centerY !== undefined) {
        const scaleRatio = newScale / viewTransform.scale;
        newX = centerX - (centerX - viewTransform.x) * scaleRatio;
        newY = centerY - (centerY - viewTransform.y) * scaleRatio;
      }

      set((state) => ({
        viewTransform: { scale: newScale, x: newX, y: newY },
        viewTransformVersion: state.viewTransformVersion + 1,
      }));
    },

    pan: (deltaX, deltaY) => {
      const { viewTransform } = get();
      set((state) => ({
        viewTransform: {
          ...viewTransform,
          x: viewTransform.x + deltaX,
          y: viewTransform.y + deltaY,
        },
        viewTransformVersion: state.viewTransformVersion + 1,
      }));
    },

    resetView: () => {
      set((state) => ({
        viewTransform: { ...DEFAULT_VIEW_TRANSFORM },
        viewTransformVersion: state.viewTransformVersion + 1,
      }));
    },

    // Annotation mode
    setAnnotationMode: (mode) => {
      set({ annotationMode: mode });
    },

    toggleAnnotationMode: (mode) => {
      const { annotationMode } = get();
      set({ annotationMode: annotationMode === mode ? 'none' : mode });
    },

    // Reset all state
    reset: () => {
      set({
        currentFrame: 0,
        playbackState: 'stopped',
        playbackSpeed: 1.0,
        isLooping: true,
        totalFrames: 0,
        frameRate: 15,
        viewTransform: { ...DEFAULT_VIEW_TRANSFORM },
        viewTransformVersion: 0,
        annotationMode: 'none',
      });
    },
  }))
);

// Selectors for optimized subscriptions
export const selectCurrentFrame = (state: PlayerState) => state.currentFrame;
export const selectPlaybackState = (state: PlayerState) => state.playbackState;
export const selectViewTransform = (state: PlayerState) => state.viewTransform;
export const selectViewTransformVersion = (state: PlayerState) => state.viewTransformVersion;
export const selectAnnotationMode = (state: PlayerState) => state.annotationMode;
