/**
 * Coronary RWS Analyser - Main Application Component
 *
 * This is the root component of the application.
 * It provides the main layout structure:
 * - Header with menu and controls
 * - Left sidebar with tools
 * - Main viewer area (canvas layers)
 * - Right panel with data displays (ECG, QCA, RWS)
 * - Bottom panel with timeline/playback controls
 */

function App() {
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4">
        <h1 className="text-lg font-semibold">Coronary RWS Analyser</h1>
        <span className="ml-2 text-xs text-gray-400">v1.0.0</span>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Tools */}
        <aside className="w-16 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4 gap-4">
          <div className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-gray-400 hover:bg-gray-600 cursor-pointer">
            {/* File icon placeholder */}
            <span className="text-xs">File</span>
          </div>
          <div className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-gray-400 hover:bg-gray-600 cursor-pointer">
            {/* Seed point icon placeholder */}
            <span className="text-xs">Seed</span>
          </div>
          <div className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-gray-400 hover:bg-gray-600 cursor-pointer">
            {/* ROI icon placeholder */}
            <span className="text-xs">ROI</span>
          </div>
        </aside>

        {/* Main Viewer Area */}
        <main className="flex-1 flex flex-col">
          {/* Canvas Container */}
          <div className="flex-1 bg-black flex items-center justify-center">
            <div className="text-gray-500 text-center">
              <p className="text-xl mb-2">Coronary RWS Analyser</p>
              <p className="text-sm">Open a DICOM file to begin analysis</p>
              <p className="text-xs mt-4 text-gray-600">
                File → Open DICOM or drag and drop
              </p>
            </div>
          </div>

          {/* Timeline / Playback Controls */}
          <div className="h-24 bg-gray-800 border-t border-gray-700 p-2">
            <div className="h-full flex flex-col">
              {/* Frame slider placeholder */}
              <div className="flex-1 bg-gray-700 rounded mb-2"></div>
              {/* Playback buttons placeholder */}
              <div className="flex items-center justify-center gap-4">
                <button className="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600 text-sm">
                  ⏮ Prev
                </button>
                <button className="px-4 py-1 bg-primary-600 rounded hover:bg-primary-500 text-sm">
                  ▶ Play
                </button>
                <button className="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600 text-sm">
                  Next ⏭
                </button>
                <span className="text-gray-400 text-sm ml-4">
                  Frame: 0 / 0
                </span>
              </div>
            </div>
          </div>
        </main>

        {/* Right Panel - Data Displays */}
        <aside className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col">
          {/* ECG Panel */}
          <div className="h-32 border-b border-gray-700 p-2">
            <div className="text-xs text-gray-400 mb-1">ECG</div>
            <div className="h-20 bg-gray-900 rounded flex items-center justify-center text-gray-600 text-xs">
              ECG waveform will appear here
            </div>
          </div>

          {/* QCA Panel */}
          <div className="flex-1 border-b border-gray-700 p-2 overflow-auto">
            <div className="text-xs text-gray-400 mb-2">QCA Metrics</div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">MLD:</span>
                <span>-- mm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Proximal RD:</span>
                <span>-- mm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Distal RD:</span>
                <span>-- mm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">DS%:</span>
                <span>-- %</span>
              </div>
            </div>
          </div>

          {/* RWS Panel - Primary Feature */}
          <div className="h-48 p-2">
            <div className="text-xs text-gray-400 mb-2">
              RWS (Radial Wall Strain)
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">MLD RWS:</span>
                <span className="text-lg font-semibold text-vessel-mld">
                  -- %
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Proximal RWS:</span>
                <span>-- %</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Distal RWS:</span>
                <span>-- %</span>
              </div>
              <div className="mt-4 p-2 bg-gray-900 rounded text-xs text-gray-500">
                Normal: &lt;8% | Intermediate: 8-12% | High: &gt;12%
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* Status Bar */}
      <footer className="h-6 bg-gray-800 border-t border-gray-700 flex items-center px-4 text-xs text-gray-400">
        <span>Ready</span>
        <span className="ml-auto">Backend: Disconnected</span>
      </footer>
    </div>
  );
}

export default App;
