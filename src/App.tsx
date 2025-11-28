/**
 * Coronary RWS Analyser - Main Application Component
 *
 * Root component providing the main layout:
 * - Header with title and status
 * - Left toolbar with tools
 * - Main viewer area (multi-layer canvas)
 * - Right panel with data displays (ECG, QCA, RWS, Tracking, Calibration, Export)
 * - Bottom panel with playback controls
 */

import { useCallback, useRef, useState } from 'react';
import { useDicomStore, useEcgStore, useCalibrationStore } from './stores';
import { VideoPlayer } from './components/Viewer';
import { PlaybackControls, Toolbar } from './components/Controls';
import {
  ECGPanel,
  QCAPanel,
  RWSPanel,
  SegmentationPanel,
  CalibrationPanel,
  ExportPanel,
  TrackingPanel,
  MetadataDisplay,
  FrameRangeSelection,
} from './components/Panels';
import { Badge, KeyboardShortcuts, SessionManager } from './components/common';

type RightPanelTab = 'analysis' | 'tracking' | 'settings' | 'info';

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<RightPanelTab>('analysis');

  const loadFile = useDicomStore((s) => s.loadFile);
  const isLoading = useDicomStore((s) => s.isLoading);
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const error = useDicomStore((s) => s.error);
  const metadata = useDicomStore((s) => s.metadata);

  const loadEcg = useEcgStore((s) => s.loadEcg);
  const setFromDicom = useCalibrationStore((s) => s.setFromDicom);

  // Handle file open
  const handleFileOpen = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Handle file selection
  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      try {
        await loadFile(file);
        // Load ECG and calibration after DICOM loads
        loadEcg();
        setFromDicom();
      } catch (error) {
        console.error('Failed to load DICOM:', error);
      }

      // Reset input
      e.target.value = '';
    },
    [loadFile, loadEcg, setFromDicom]
  );

  // Handle drag and drop
  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (!file) return;

      try {
        await loadFile(file);
        loadEcg();
        setFromDicom();
      } catch (error) {
        console.error('Failed to load DICOM:', error);
      }
    },
    [loadFile, loadEcg, setFromDicom]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const tabs: { id: RightPanelTab; label: string }[] = [
    { id: 'analysis', label: 'Analysis' },
    { id: 'tracking', label: 'Tracking' },
    { id: 'settings', label: 'Settings' },
    { id: 'info', label: 'Info' },
  ];

  return (
    <div
      className="flex flex-col h-screen bg-gray-900 text-white"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      {/* Session Manager (auto-save recovery) */}
      <SessionManager />

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".dcm,.dicom,application/dicom"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Header */}
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4">
        <h1 className="text-lg font-semibold">Coronary RWS Analyser</h1>
        <span className="text-xs text-gray-400">v1.0.0</span>

        {/* Loading indicator */}
        {isLoading && (
          <Badge variant="info">Loading...</Badge>
        )}

        {/* Error indicator */}
        {error && (
          <Badge variant="danger">{error}</Badge>
        )}

        {/* File info */}
        {isLoaded && metadata && (
          <span className="text-xs text-gray-400 ml-auto">
            {metadata.numFrames} frames | {metadata.rows}x{metadata.columns}
            {metadata.pixelSpacing && (
              <span className="ml-2">
                | {metadata.pixelSpacing[0].toFixed(4)} mm/px
              </span>
            )}
          </span>
        )}
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Toolbar */}
        <Toolbar onFileOpen={handleFileOpen} />

        {/* Main Viewer Area */}
        <main className="flex-1 flex flex-col">
          {/* Video Player */}
          <div className="flex-1 relative">
            <VideoPlayer />
          </div>

          {/* Playback Controls */}
          <div className="h-24 bg-gray-800 border-t border-gray-700">
            <PlaybackControls />
          </div>
        </main>

        {/* Right Panel - Tabbed Interface */}
        <aside className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-gray-700">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === 'analysis' && (
              <div className="flex flex-col h-full">
                {/* ECG Panel */}
                <div className="h-24 border-b border-gray-700 flex-shrink-0">
                  <ECGPanel />
                </div>

                {/* Segmentation Panel */}
                <div className="p-3 border-b border-gray-700">
                  <SegmentationPanel />
                </div>

                {/* QCA Panel */}
                <div className="p-3 border-b border-gray-700">
                  <QCAPanel />
                </div>

                {/* RWS Panel - Primary Feature */}
                <div className="p-3 border-b border-gray-700">
                  <FrameRangeSelection />
                </div>

                <div className="p-3 flex-1">
                  <RWSPanel />
                </div>
              </div>
            )}

            {activeTab === 'tracking' && (
              <div className="p-3">
                <TrackingPanel />
              </div>
            )}

            {activeTab === 'settings' && (
              <div className="p-3 space-y-4">
                <CalibrationPanel />
                <div className="border-t border-gray-700 pt-4">
                  <ExportPanel />
                </div>
              </div>
            )}

            {activeTab === 'info' && (
              <div className="p-3 space-y-4">
                <MetadataDisplay />
                <div className="border-t border-gray-700 pt-4">
                  <KeyboardShortcuts />
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Status Bar */}
      <footer className="h-6 bg-gray-800 border-t border-gray-700 flex items-center px-4 text-xs text-gray-400">
        <span>{isLoaded ? 'Ready' : 'No file loaded'}</span>
        <span className="ml-4">
          Press <kbd className="bg-gray-700 px-1 rounded">B</kbd> for ROI,{' '}
          <kbd className="bg-gray-700 px-1 rounded">S</kbd> for Seed,{' '}
          <kbd className="bg-gray-700 px-1 rounded">?</kbd> for shortcuts
        </span>
        <span className="ml-auto flex items-center gap-2">
          <KeyboardShortcuts compact />
          Backend:{' '}
          <span className="text-green-400">Connected</span>
        </span>
      </footer>
    </div>
  );
}

export default App;
