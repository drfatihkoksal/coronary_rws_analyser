/**
 * Development startup script
 *
 * Starts both the Python backend and Tauri frontend in development mode.
 * The backend runs in a separate terminal window.
 */

import { spawn, exec } from "child_process";
import { platform } from "os";

const isWindows = platform() === "win32";
const isWSL = process.env.WSL_DISTRO_NAME !== undefined;

console.log("🚀 Starting Coronary RWS Analyser in development mode...\n");

// Start Python backend
console.log("📦 Starting Python backend...");

let backendProcess;

if (isWindows) {
  // Windows: Open new cmd window
  backendProcess = spawn("cmd", ["/c", "start", "cmd", "/k", "cd python-backend && python -m uvicorn app.main:app --reload --port 8000"], {
    shell: true,
    detached: true,
  });
} else if (isWSL) {
  // WSL: Use gnome-terminal or xterm if available, otherwise run in background
  exec("which gnome-terminal", (error) => {
    if (!error) {
      backendProcess = spawn("gnome-terminal", ["--", "bash", "-c", "cd python-backend && source venv/bin/activate && python -m uvicorn app.main:app --reload --port 8000; exec bash"], {
        detached: true,
      });
    } else {
      // Fallback: run in background
      backendProcess = spawn("bash", ["-c", "cd python-backend && source venv/bin/activate && python -m uvicorn app.main:app --reload --port 8000"], {
        detached: true,
        stdio: "ignore",
      });
    }
  });
} else {
  // Linux/Mac: Use x-terminal-emulator or run in background
  backendProcess = spawn("bash", ["-c", "cd python-backend && source venv/bin/activate && python -m uvicorn app.main:app --reload --port 8000"], {
    detached: true,
    stdio: "ignore",
  });
}

// Wait a moment for backend to start
setTimeout(() => {
  console.log("✅ Backend starting on http://127.0.0.1:8000");
  console.log("\n🖥️  Starting Tauri frontend...\n");

  // Start Tauri dev
  const tauriProcess = spawn("npm", ["run", "tauri", "dev"], {
    stdio: "inherit",
    shell: true,
  });

  tauriProcess.on("close", (code) => {
    console.log(`\n👋 Tauri exited with code ${code}`);
    console.log("💡 Remember to stop the Python backend manually if needed");
    process.exit(code);
  });

}, 2000);

// Handle cleanup
process.on("SIGINT", () => {
  console.log("\n🛑 Shutting down...");
  process.exit(0);
});
