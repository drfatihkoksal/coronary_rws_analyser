/**
 * ECG Panel Component
 *
 * Displays ECG waveform with R-peak markers.
 */

import { useEffect, useRef } from 'react';
import { useEcgStore, usePlayerStore } from '../../stores';
import { Card } from '../common/Card';

export function ECGPanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const ecgData = useEcgStore((s) => s.ecgData);
  const hasEcg = useEcgStore((s) => s.hasEcg);
  const heartRate = useEcgStore((s) => s.heartRate);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const frameRate = usePlayerStore((s) => s.frameRate);

  // Render ECG waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !ecgData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw ECG signal
    const signal = ecgData.signal;
    if (signal.length === 0) return;

    // Normalize signal
    const min = Math.min(...signal);
    const max = Math.max(...signal);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1.5;

    const step = width / signal.length;
    for (let i = 0; i < signal.length; i++) {
      const x = i * step;
      const y = height - ((signal[i] - min) / range) * (height - 20) - 10;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw R-peaks
    if (ecgData.rPeaks) {
      ctx.fillStyle = '#ef4444';
      ecgData.rPeaks.forEach((peak) => {
        const x = (peak / signal.length) * width;
        ctx.beginPath();
        ctx.arc(x, 10, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw current frame indicator
    if (totalFrames > 0 && frameRate > 0) {
      const currentTime = currentFrame / frameRate;
      const ecgDuration = ecgData.duration;
      const x = (currentTime / ecgDuration) * width;

      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }, [ecgData, currentFrame, totalFrames, frameRate]);

  if (!hasEcg) {
    return (
      <Card title="ECG" className="h-full">
        <div className="h-20 flex items-center justify-center text-gray-500 text-sm">
          No ECG data available
        </div>
      </Card>
    );
  }

  return (
    <Card title="ECG" className="h-full" noPadding>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={300}
          height={80}
          className="w-full"
        />

        {/* Heart rate overlay */}
        {heartRate && (
          <div className="absolute top-1 right-1 bg-black/50 px-1.5 py-0.5 rounded text-xs">
            <span className="text-red-400">♥</span>
            <span className="text-white ml-1">{Math.round(heartRate)} BPM</span>
          </div>
        )}
      </div>
    </Card>
  );
}
