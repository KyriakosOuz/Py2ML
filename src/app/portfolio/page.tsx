'use client';

import { useState, useEffect } from 'react';
import { Trophy, Calendar, Tag } from 'lucide-react';
import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import type { ProjectData } from '@/types';

export default function PortfolioPage() {
  const [projects, setProjects] = useState<ProjectData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/projects')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setProjects(data.filter((p: ProjectData) => p.status === 'COMPLETED'));
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surface rounded w-48" />
          {[1, 2].map((i) => (
            <div key={i} className="h-32 bg-surface rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Trophy size={28} className="text-primary" />
          <h1 className="font-serif text-3xl text-text-primary">Portfolio</h1>
        </div>
        <p className="text-text-secondary">
          Your completed projects showcasing your Python and ML skills.
        </p>
      </div>

      {projects.length === 0 ? (
        <Card>
          <div className="text-center py-8">
            <Trophy size={48} className="text-text-muted mx-auto mb-4" />
            <h3 className="font-serif text-lg text-text-primary mb-2">No completed projects yet</h3>
            <p className="text-text-secondary text-sm">
              Complete projects to build your portfolio!
            </p>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {projects.map((project) => (
            <Card key={project.id}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-serif text-lg text-text-primary mb-1">{project.title}</h3>
                  <p className="text-text-secondary text-sm mb-3">{project.brief}</p>
                  <div className="flex items-center gap-3">
                    <Badge variant="success" size="sm">Completed</Badge>
                    <div className="flex items-center gap-1 text-xs text-text-muted">
                      <Calendar size={12} />
                      <span>Recently</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1 text-xs text-text-muted">
                  <Tag size={12} />
                  <span>{project.stage}</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
