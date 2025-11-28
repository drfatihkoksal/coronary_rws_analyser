/**
 * Slider Component
 *
 * Range input with custom styling.
 */

import { InputHTMLAttributes } from 'react';

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  showValue?: boolean;
}

export function Slider({ label, showValue = false, className = '', ...props }: SliderProps) {
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs text-gray-400">
          {label && <span>{label}</span>}
          {showValue && <span>{props.value}</span>}
        </div>
      )}
      <input
        type="range"
        className="
          w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:bg-blue-500
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:hover:bg-blue-400
        "
        {...props}
      />
    </div>
  );
}
