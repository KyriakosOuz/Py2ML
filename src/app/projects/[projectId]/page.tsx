'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import ProjectDetail from '@/components/projects/ProjectDetail';
import Button from '@/components/ui/Button';
import type { ProjectData } from '@/types';

export default function ProjectDetailPage() {
  const params = useParams();
  const projectId = params.projectId as string;
  const [project, setProject] = useState<ProjectData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!projectId) return;
    fetch(`/api/projects/${projectId}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setProject({
            ...data,
            requirements: data.requirements || [],
            stretchGoals: data.stretchGoals || [],
            steps: data.steps || [],
            rubric: data.rubric || [],
          });
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [projectId]);

  const handleStatusChange = async (status: string) => {
    try {
      await fetch(`/api/projects/${projectId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      setProject((prev) => prev ? { ...prev, status } : prev);
    } catch {
      // handle silently
    }
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-surface rounded w-64 mb-8" />
          <div className="grid lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-32 bg-surface rounded-xl" />
              ))}
            </div>
            <div className="h-96 bg-surface rounded-xl" />
          </div>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <h1 className="font-serif text-2xl text-text-primary mb-4">Project Not Found</h1>
        <Link href="/projects">
          <Button variant="secondary">Back to Projects</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <ProjectDetail project={project} onStatusChange={handleStatusChange} />
    </div>
  );
}
