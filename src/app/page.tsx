'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import { ArrowRight, BookOpen, Code2, Brain, Rocket, Bot, Briefcase } from 'lucide-react';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';

export default function HomePage() {
  const { data: session } = useSession();
  const [nextLesson, setNextLesson] = useState<{ id: string; title: string } | null>(null);

  useEffect(() => {
    if (!session?.user) return;
    fetch('/api/dashboard')
      .then((res) => {
        if (!res.ok) return null;
        return res.json();
      })
      .then((data) => {
        if (data?.nextLesson) setNextLesson(data.nextLesson);
      })
      .catch(console.error);
  }, [session]);

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
      title: 'Data Structures & Logic',
      description: 'Learn NumPy, Pandas, and Matplotlib for data analysis and visualization.',
      lessons: 7,
      color: 'text-green-400',
    },
    {
      icon: Brain,
      title: 'ML & Deep Learning',
      description: 'Build ML models with scikit-learn, then dive into neural networks with TensorFlow.',
      lessons: 12,
      color: 'text-purple-400',
    },
    {
      icon: Bot,
      title: 'NLP & Agentic AI',
      description: 'Master transformers, prompt engineering, RAG pipelines, and autonomous AI agents.',
      lessons: 13,
      color: 'text-cyan-400',
    },
    {
      icon: Briefcase,
      title: 'MLOps & Career Ready',
      description: 'Deploy models with FastAPI & Docker. Build your portfolio and ace interviews.',
      lessons: 7,
      color: 'text-amber-400',
    },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-1.5 rounded-full text-sm mb-6">
          <Rocket size={16} />
          From Zero to AI Engineer
        </div>
        <h1 className="font-serif text-4xl md:text-5xl text-text-primary mb-4">
          Learn Python.<br />
          <span className="text-primary">Build Intelligence.</span>
        </h1>
        <p className="text-text-secondary text-lg max-w-2xl mx-auto mb-8">
          A structured curriculum that takes you from your first <code className="text-primary bg-surface-light px-1.5 py-0.5 rounded text-sm font-mono">print()</code> to
          building AI agents and deploying ML models. 52 lessons, real projects, and instant feedback.
        </p>
        <div className="flex items-center justify-center gap-4">
          {session ? (
            <Link href={nextLesson ? `/lesson/${nextLesson.id}` : '/curriculum'}>
              <Button size="lg">
                {nextLesson ? 'Continue Learning' : 'Start Learning'}
                <ArrowRight size={18} className="ml-2" />
              </Button>
            </Link>
          ) : (
            <Link href="/signup">
              <Button size="lg">
                Get Started Free
                <ArrowRight size={18} className="ml-2" />
              </Button>
            </Link>
          )}
          <Link href={session ? '/curriculum' : '/login'}>
            <Button variant="secondary" size="lg">
              {session ? 'View Curriculum' : 'Sign In'}
            </Button>
          </Link>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {[
          { label: 'Lessons', value: '52' },
          { label: 'Exercises', value: '150+' },
          { label: 'Quizzes', value: '150+' },
          { label: 'Projects', value: '16' },
        ].map((stat) => (
          <div key={stat.label} className="text-center p-4 bg-surface rounded-xl border border-border">
            <p className="font-mono text-2xl text-primary font-bold">{stat.value}</p>
            <p className="text-text-muted text-sm">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Stages */}
      <h2 className="font-serif text-2xl text-text-primary mb-6">Your Path to AI Mastery</h2>
      <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-4 mb-16">
        {stages.map((stage, i) => (
          <Card key={i} hover>
            <stage.icon className={`${stage.color} mb-3`} size={28} />
            <h3 className="font-serif text-base text-text-primary mb-1">{stage.title}</h3>
            <p className="text-text-secondary text-xs mb-3 leading-relaxed">{stage.description}</p>
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
          { title: 'Real Projects', desc: 'Apply your skills with 16 hands-on projects from CLI tools to deployed AI agents.' },
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
