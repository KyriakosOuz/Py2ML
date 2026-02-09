'use client';

import Badge from '@/components/ui/Badge';

interface SkillTagsProps {
  skills: { name: string; slug: string; earnedAt: string }[];
}

export default function SkillTags({ skills }: SkillTagsProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-6">
      <h3 className="font-serif text-lg text-text-primary mb-4">Skills Earned</h3>
      {skills.length === 0 ? (
        <p className="text-text-muted text-sm">Complete exercises to earn skill badges!</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {skills.map((skill) => (
            <Badge key={skill.slug} variant="success" size="md">
              {skill.name}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
