import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0f1117",
        surface: "#1a1d27",
        "surface-light": "#252836",
        primary: "#f59e0b",
        "primary-hover": "#d97706",
        success: "#10b981",
        error: "#ef4444",
        warning: "#f59e0b",
        "text-primary": "#e2e8f0",
        "text-secondary": "#94a3b8",
        "text-muted": "#64748b",
        border: "#2d3148",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        serif: ["DM Serif Display", "serif"],
        sans: ["DM Sans", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
