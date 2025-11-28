/**
 * Keyboard Shortcuts Component
 *
 * Displays available keyboard shortcuts.
 */

import { useState } from 'react';

interface ShortcutGroup {
  title: string;
  shortcuts: { key: string; description: string }[];
}

const SHORTCUT_GROUPS: ShortcutGroup[] = [
  {
    title: 'Playback',
    shortcuts: [
      { key: 'Space', description: 'Play / Pause' },
      { key: '\u2190 / \u2192', description: 'Previous / Next frame' },
      { key: 'Home / End', description: 'First / Last frame' },
    ],
  },
  {
    title: 'View',
    shortcuts: [
      { key: 'Scroll / +/-', description: 'Zoom in / out' },
      { key: '0', description: 'Reset zoom' },
      { key: 'F', description: 'Fit to window' },
    ],
  },
  {
    title: 'Tools',
    shortcuts: [
      { key: 'B', description: 'Draw ROI box' },
      { key: 'S', description: 'Place seed points' },
      { key: 'V', description: 'Select tool' },
      { key: 'H', description: 'Pan tool' },
      { key: 'Escape', description: 'Cancel / Deselect' },
    ],
  },
  {
    title: 'Actions',
    shortcuts: [
      { key: 'Enter', description: 'Run segmentation' },
      { key: 'Ctrl+S', description: 'Save session' },
      { key: 'Ctrl+O', description: 'Open DICOM file' },
      { key: 'Ctrl+E', description: 'Export results' },
    ],
  },
];

interface KeyboardShortcutsProps {
  compact?: boolean;
}

export function KeyboardShortcuts({ compact = false }: KeyboardShortcutsProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (compact) {
    return (
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="text-gray-400 hover:text-white transition-colors text-sm"
          title="Keyboard shortcuts"
        >
          <span className="text-lg">?</span>
        </button>

        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Popup */}
            <div className="absolute bottom-full right-0 mb-2 w-72 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 p-4">
              <h3 className="text-white font-medium mb-3">Keyboard Shortcuts</h3>
              <div className="space-y-3 text-xs">
                {SHORTCUT_GROUPS.map((group) => (
                  <div key={group.title}>
                    <h4 className="text-gray-400 font-medium mb-1">{group.title}</h4>
                    <div className="space-y-1">
                      {group.shortcuts.map((shortcut) => (
                        <div key={shortcut.key} className="flex justify-between">
                          <kbd className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 font-mono">
                            {shortcut.key}
                          </kbd>
                          <span className="text-gray-400">{shortcut.description}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-white">Keyboard Shortcuts</h3>

      {SHORTCUT_GROUPS.map((group) => (
        <div key={group.title} className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300 border-b border-gray-700 pb-1">
            {group.title}
          </h4>
          <div className="space-y-1">
            {group.shortcuts.map((shortcut) => (
              <div
                key={shortcut.key}
                className="flex items-center justify-between text-sm"
              >
                <kbd className="bg-gray-700 px-2 py-1 rounded text-gray-300 font-mono text-xs">
                  {shortcut.key}
                </kbd>
                <span className="text-gray-400">{shortcut.description}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
