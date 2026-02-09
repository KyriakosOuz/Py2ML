interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export default function Card({ children, className = '', hover = false, onClick }: CardProps) {
  return (
    <div
      className={`bg-surface border border-border rounded-xl p-6 ${
        hover ? 'hover:border-primary/50 hover:bg-surface-light transition-all cursor-pointer' : ''
      } ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
}
