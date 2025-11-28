/**
 * Toolbar Component
 *
 * Left sidebar toolbar with tool buttons.
 * Synced with AnnotationCanvas via playerStore.annotationMode
 */

import { useDicomStore, usePlayerStore, type AnnotationMode } from '../../stores';
import { Tooltip } from '../common/Tooltip';

interface ToolbarProps {
  onFileOpen?: () => void;
}

export function Toolbar({ onFileOpen }: ToolbarProps) {
  const isLoaded = useDicomStore((s) => s.isLoaded);
  const annotationMode = usePlayerStore((s) => s.annotationMode);
  const toggleAnnotationMode = usePlayerStore((s) => s.toggleAnnotationMode);

  const tools: { id: AnnotationMode; icon: string; label: string; shortcut: string }[] = [
    { id: 'select', icon: '👆', label: 'Select', shortcut: 'V' },
    { id: 'roi', icon: '⬜', label: 'Draw ROI', shortcut: 'B' },
    { id: 'fixed-roi', icon: '🎯', label: 'Fixed ROI (150×150)', shortcut: 'F' },
    { id: 'seed', icon: '📍', label: 'Place Seed', shortcut: 'S' },
    { id: 'pan', icon: '✋', label: 'Pan', shortcut: 'H' },
  ];

  const handleToolClick = (toolId: AnnotationMode) => {
    if (!isLoaded) return;
    toggleAnnotationMode(toolId);
  };

  return (
    <aside className="w-14 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-3 gap-2">
      {/* File button */}
      <Tooltip content="Open DICOM file (Ctrl+O)" position="right">
        <button
          onClick={onFileOpen}
          className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-gray-300 hover:bg-gray-600 hover:text-white transition-colors"
        >
          📁
        </button>
      </Tooltip>

      {/* Divider */}
      <div className="w-8 h-px bg-gray-700 my-1" />

      {/* Tools */}
      {tools.map((tool) => (
        <Tooltip key={tool.id} content={`${tool.label} (${tool.shortcut})`} position="right">
          <button
            onClick={() => handleToolClick(tool.id)}
            disabled={!isLoaded}
            className={`
              w-10 h-10 rounded-lg flex items-center justify-center transition-colors
              ${
                annotationMode === tool.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
              }
              ${!isLoaded ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            {tool.icon}
          </button>
        </Tooltip>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Settings */}
      <Tooltip content="Settings" position="right">
        <button className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-gray-300 hover:bg-gray-600 hover:text-white transition-colors">
          ⚙️
        </button>
      </Tooltip>
    </aside>
  );
}
