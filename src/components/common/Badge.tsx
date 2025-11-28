/**
 * Badge Component
 *
 * Small status indicator.
 */

import { ReactNode } from 'react';

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info';

interface BadgeProps {
  variant?: BadgeVariant;
  children: ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-gray-600 text-gray-200',
  success: 'bg-green-600 text-green-100',
  warning: 'bg-amber-600 text-amber-100',
  danger: 'bg-red-600 text-red-100',
  info: 'bg-blue-600 text-blue-100',
};

export function Badge({ variant = 'default', children, className = '' }: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center px-2 py-0.5 rounded text-xs font-medium
        ${variantStyles[variant]}
        ${className}
      `}
    >
      {children}
    </span>
  );
}
