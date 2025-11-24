/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Medical imaging dark theme
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        // Vessel visualization colors
        vessel: {
          segmentation: '#22c55e',  // Green for segmented vessel
          centerline: '#f59e0b',     // Amber for centerline
          mld: '#ef4444',            // Red for MLD marker
          reference: '#3b82f6',      // Blue for reference diameters
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
      },
    },
  },
  plugins: [],
}
