'use client';

interface ProgressBarProps {
  value: number;
  max: number;
  label?: string;
  showPercent?: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'success' | 'error';
}

const sizeClasses = {
  sm: 'h-1.5',
  md: 'h-2.5',
  lg: 'h-4',
};

const colorClasses = {
  primary: 'bg-primary',
  success: 'bg-success',
  error: 'bg-error',
};

export default function ProgressBar({
  value,
  max,
  label,
  showPercent = false,
  size = 'md',
  color = 'primary',
}: ProgressBarProps) {
  const percent = max > 0 ? Math.round((value / max) * 100) : 0;

  return (
    <div className="w-full">
      {(label || showPercent) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-sm text-text-secondary">{label}</span>}
          {showPercent && (
            <span className="text-sm font-mono text-text-muted">{percent}%</span>
          )}
        </div>
      )}
      <div className={`w-full bg-surface-light rounded-full ${sizeClasses[size]} overflow-hidden`}>
        <div
          className={`${colorClasses[color]} ${sizeClasses[size]} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
