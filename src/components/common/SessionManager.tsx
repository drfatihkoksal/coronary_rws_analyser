/**
 * Session Manager Component
 *
 * Handles auto-save and session recovery.
 */

import { useEffect, useState, useCallback } from 'react';
import { usePlayerStore, useDicomStore, useSegmentationStore, useAnnotationStore } from '../../stores';

const SESSION_KEY = 'coronary_rws_session';
const AUTO_SAVE_INTERVAL = 30000; // 30 seconds

interface SessionData {
  timestamp: number;
  dicomPath?: string;
  currentFrame: number;
  annotationMode: string;
  seedPoints: Record<number, { x: number; y: number }[]>;
  roi: Record<number, { x: number; y: number; width: number; height: number } | null>;
}

interface SessionManagerProps {
  onRestore?: () => void;
}

export function SessionManager({ onRestore }: SessionManagerProps) {
  const [hasRecoverableSession, setHasRecoverableSession] = useState(false);
  const [showRecoveryDialog, setShowRecoveryDialog] = useState(false);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const annotationMode = usePlayerStore((s) => s.annotationMode);
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const frameAnnotations = useAnnotationStore((s) => s.frameAnnotations);

  // Check for recoverable session on mount
  useEffect(() => {
    const savedSession = localStorage.getItem(SESSION_KEY);
    if (savedSession) {
      try {
        const session: SessionData = JSON.parse(savedSession);
        const age = Date.now() - session.timestamp;

        // Only recover if session is less than 24 hours old
        if (age < 24 * 60 * 60 * 1000) {
          setHasRecoverableSession(true);
          setShowRecoveryDialog(true);
        } else {
          localStorage.removeItem(SESSION_KEY);
        }
      } catch {
        localStorage.removeItem(SESSION_KEY);
      }
    }
  }, []);

  // Auto-save session
  useEffect(() => {
    if (!isLoaded) return;

    const saveSession = () => {
      const seedPointsObj: Record<number, { x: number; y: number }[]> = {};
      const roiObj: Record<number, { x: number; y: number; width: number; height: number } | null> = {};

      frameAnnotations.forEach((annotation, frameIndex) => {
        if (annotation.seedPoints.length > 0) {
          seedPointsObj[frameIndex] = annotation.seedPoints;
        }
        if (annotation.roi) {
          roiObj[frameIndex] = annotation.roi;
        }
      });

      const session: SessionData = {
        timestamp: Date.now(),
        currentFrame,
        annotationMode,
        seedPoints: seedPointsObj,
        roi: roiObj,
      };

      localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    };

    // Save immediately
    saveSession();

    // Set up auto-save interval
    const interval = setInterval(saveSession, AUTO_SAVE_INTERVAL);

    return () => clearInterval(interval);
  }, [isLoaded, currentFrame, annotationMode, frameAnnotations]);

  // Recover session
  const recoverSession = useCallback(() => {
    const savedSession = localStorage.getItem(SESSION_KEY);
    if (!savedSession) return;

    try {
      const session: SessionData = JSON.parse(savedSession);

      // Restore frame position
      usePlayerStore.getState().setCurrentFrame(session.currentFrame);

      // Restore annotation mode
      if (session.annotationMode !== 'none') {
        usePlayerStore.getState().setAnnotationMode(session.annotationMode as any);
      }

      // Restore annotations
      const annotationStore = useAnnotationStore.getState();
      Object.entries(session.seedPoints).forEach(([frameStr, points]) => {
        const frameIndex = parseInt(frameStr);
        points.forEach((point) => {
          annotationStore.addSeedPoint(frameIndex, point);
        });
      });

      Object.entries(session.roi).forEach(([frameStr, roi]) => {
        const frameIndex = parseInt(frameStr);
        if (roi) {
          annotationStore.setROI(frameIndex, roi);
        }
      });

      onRestore?.();
    } catch (e) {
      console.error('Failed to recover session:', e);
    }

    setShowRecoveryDialog(false);
    setHasRecoverableSession(false);
  }, [onRestore]);

  // Discard session
  const discardSession = useCallback(() => {
    localStorage.removeItem(SESSION_KEY);
    setShowRecoveryDialog(false);
    setHasRecoverableSession(false);
  }, []);

  // Clear session on successful completion
  const clearSession = useCallback(() => {
    localStorage.removeItem(SESSION_KEY);
  }, []);

  if (!showRecoveryDialog || !hasRecoverableSession) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
        <h3 className="text-lg font-medium text-white mb-2">Recover Previous Session?</h3>

        <p className="text-gray-400 text-sm mb-4">
          A previous session was found. Would you like to restore it?
        </p>

        <div className="flex gap-3 justify-end">
          <button
            onClick={discardSession}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Discard
          </button>
          <button
            onClick={recoverSession}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
          >
            Restore
          </button>
        </div>
      </div>
    </div>
  );
}

// Utility to manually clear session
export function clearSavedSession() {
  localStorage.removeItem(SESSION_KEY);
}
