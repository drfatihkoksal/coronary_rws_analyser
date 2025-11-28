/**
 * Card Component
 *
 * Container with optional header and padding.
 */

import { ReactNode } from 'react';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

export function Card({ title, children, className = '', noPadding = false }: CardProps) {
  return (
    <div className={`bg-gray-800 rounded-lg border border-gray-700 ${className}`}>
      {title && (
        <div className="px-3 py-2 border-b border-gray-700">
          <h3 className="text-sm font-medium text-gray-300">{title}</h3>
        </div>
      )}
      <div className={noPadding ? '' : 'p-3'}>{children}</div>
    </div>
  );
}
