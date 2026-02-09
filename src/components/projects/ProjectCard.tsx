'use client';

import Link from 'next/link';
import { FolderKanban, CheckCircle2, Clock, Circle } from 'lucide-react';
import Badge from '@/components/ui/Badge';
import type { ProjectData } from '@/types';

interface ProjectCardProps {
  project: ProjectData;
}

const stageBadgeVariant: Record<string, 'default' | 'success' | 'warning'> = {
  BEGINNER: 'default',
  DATA: 'success',
  ML: 'warning',
};

const stageLabel: Record<string, string> = {
  BEGINNER: 'Beginner',
  DATA: 'Data Science',
  ML: 'Machine Learning',
};

const statusIcons = {
  NOT_STARTED: Circle,
  IN_PROGRESS: Clock,
  COMPLETED: CheckCircle2,
};

export default function ProjectCard({ project }: ProjectCardProps) {
  const StatusIcon = statusIcons[project.status as keyof typeof statusIcons] || Circle;

  return (
    <Link href={`/projects/${project.id}`}>
      <div className="bg-surface border border-border rounded-xl p-6 hover:border-primary/50 hover:bg-surface-light transition-all cursor-pointer h-full flex flex-col">
        <div className="flex items-start justify-between mb-4">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
            <FolderKanban size={20} className="text-primary" />
          </div>
          <StatusIcon
            size={16}
            className={
              project.status === 'COMPLETED' ? 'text-success' :
              project.status === 'IN_PROGRESS' ? 'text-primary' :
              'text-text-muted'
            }
          />
        </div>
        <h3 className="font-serif text-lg text-text-primary mb-2">{project.title}</h3>
        <p className="text-text-secondary text-sm mb-4 flex-1">{project.brief}</p>
        <div className="flex items-center gap-2">
          <Badge variant={stageBadgeVariant[project.stage] || 'default'} size="sm">
            {stageLabel[project.stage] || project.stage}
          </Badge>
          <Badge variant="outline" size="sm">
            {project.requirements.length} requirements
          </Badge>
        </div>
      </div>
    </Link>
  );
}
