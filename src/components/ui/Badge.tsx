interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'outline';
  size?: 'sm' | 'md';
}

const variantClasses = {
  default: 'bg-surface-light text-text-primary',
  success: 'bg-success/20 text-success',
  warning: 'bg-primary/20 text-primary',
  error: 'bg-error/20 text-error',
  outline: 'border border-border text-text-secondary',
};

const sizeClasses = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-3 py-1 text-sm',
};

export default function Badge({ children, variant = 'default', size = 'sm' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${variantClasses[variant]} ${sizeClasses[size]}`}
    >
      {children}
    </span>
  );
}
