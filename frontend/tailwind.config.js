/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      colors: {
        graphite: "#080b0f",
        panel: "#111820",
        steel: "#19232d",
        line: "#273544",
        cyan: "#2dd4ff",
        pass: "#35d07f",
        fail: "#ff5c67",
        warn: "#f2b84b",
      },
    },
  },
  plugins: [],
};

