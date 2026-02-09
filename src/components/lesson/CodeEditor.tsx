'use client';

import { useCallback } from 'react';
import dynamic from 'next/dynamic';

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  height?: string;
}

export default function CodeEditor({ value, onChange, height = '250px' }: CodeEditorProps) {
  const handleChange = useCallback(
    (val: string | undefined) => {
      onChange(val ?? '');
    },
    [onChange]
  );

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <MonacoEditor
        height={height}
        language="python"
        theme="vs-dark"
        value={value}
        onChange={handleChange}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          fontFamily: 'JetBrains Mono, monospace',
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 4,
          insertSpaces: true,
          wordWrap: 'on',
          padding: { top: 12, bottom: 12 },
          renderLineHighlight: 'line',
          cursorBlinking: 'smooth',
          smoothScrolling: true,
        }}
        loading={
          <div className="h-full bg-surface flex items-center justify-center text-text-muted text-sm">
            Loading editor...
          </div>
        }
      />
    </div>
  );
}
