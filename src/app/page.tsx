'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, BookOpen, Code2, Brain, Rocket } from 'lucide-react';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';

export default function HomePage() {
  const [nextLesson, setNextLesson] = useState<{ id: string; title: string } | null>(null);

  useEffect(() => {
    fetch('/api/dashboard')
      .then((res) => res.json())
      .then((data) => {
        if (data.nextLesson) setNextLesson(data.nextLesson);
      })
      .catch(console.error);
  }, []);

  const stages = [
    {
      icon: Code2,
      title: 'Python Foundations',
      description: 'Master core Python: variables, data types, control flow, functions, and more.',
      lessons: 8,
      color: 'text-blue-400',
    },
    {
      icon: BookOpen,
      title: 'Python for Data',
      description: 'Learn NumPy, Pandas, and Matplotlib for data analysis and visualization.',
      lessons: 7,
      color: 'text-green-400',
    },
    {
      icon: Brain,
      title: 'ML/AI Track',
      description: 'Build ML models with scikit-learn: classification, regression, and more.',
      lessons: 5,
      color: 'text-purple-400',
    },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-1.5 rounded-full text-sm mb-6">
          <Rocket size={16} />
          From Zero to ML
        </div>
        <h1 className="font-serif text-4xl md:text-5xl text-text-primary mb-4">
          Learn Python.<br />
          <span className="text-primary">Build Intelligence.</span>
        </h1>
        <p className="text-text-secondary text-lg max-w-2xl mx-auto mb-8">
          A structured curriculum that takes you from your first <code className="text-primary bg-surface-light px-1.5 py-0.5 rounded text-sm font-mono">print()</code> to
          building machine learning models. Interactive exercises, real projects, and instant feedback.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link href={nextLesson ? `/lesson/${nextLesson.id}` : '/curriculum'}>
            <Button size="lg">
              {nextLesson ? 'Continue Learning' : 'Start Learning'}
              <ArrowRight size={18} className="ml-2" />
            </Button>
          </Link>
          <Link href="/curriculum">
            <Button variant="secondary" size="lg">
              View Curriculum
            </Button>
          </Link>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {[
          { label: 'Lessons', value: '20' },
          { label: 'Exercises', value: '60' },
          { label: 'Quizzes', value: '60' },
          { label: 'Projects', value: '6' },
        ].map((stat) => (
          <div key={stat.label} className="text-center p-4 bg-surface rounded-xl border border-border">
            <p className="font-mono text-2xl text-primary font-bold">{stat.value}</p>
            <p className="text-text-muted text-sm">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Stages */}
      <h2 className="font-serif text-2xl text-text-primary mb-6">Three Stages to Mastery</h2>
      <div className="grid md:grid-cols-3 gap-6 mb-16">
        {stages.map((stage, i) => (
          <Card key={i} hover>
            <stage.icon className={`${stage.color} mb-4`} size={32} />
            <h3 className="font-serif text-lg text-text-primary mb-2">{stage.title}</h3>
            <p className="text-text-secondary text-sm mb-4">{stage.description}</p>
            <p className="text-text-muted text-xs font-mono">{stage.lessons} lessons</p>
          </Card>
        ))}
      </div>

      {/* Features */}
      <h2 className="font-serif text-2xl text-text-primary mb-6">How It Works</h2>
      <div className="grid md:grid-cols-2 gap-6">
        {[
          { title: 'Interactive Code Editor', desc: 'Write and run Python directly in your browser with Monaco editor.' },
          { title: 'Instant Feedback', desc: 'Get immediate pass/fail on exercises with helpful error messages.' },
          { title: 'Progressive Hints', desc: 'Stuck? Reveal hints one at a time to guide your thinking.' },
          { title: 'Real Projects', desc: 'Apply your skills with 6 hands-on projects from CLI tools to ML models.' },
        ].map((feature, i) => (
          <div key={i} className="flex gap-4 p-4">
            <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-primary font-mono font-bold text-sm">{i + 1}</span>
            </div>
            <div>
              <h3 className="text-text-primary font-medium mb-1">{feature.title}</h3>
              <p className="text-text-secondary text-sm">{feature.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
