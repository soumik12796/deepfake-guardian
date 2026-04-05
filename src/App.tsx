import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Shield, Fingerprint, Upload, Search, AlertTriangle,
  CheckCircle2, Info, Cpu, Activity, Lock, RefreshCw, ZoomIn
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { cn } from './lib/utils';
import { analyzeImage, AnalysisResult, HeatmapMode } from './services/gemini';

// ============================================================
// DBSCAN HEATMAP ENGINE (Client-Side, No API)
// ============================================================

interface Point { x: number; y: number; value: number; }

function extractAnomalyPoints(imageData: ImageData, mode: HeatmapMode, sampleRate = 3): Point[] {
  const { data, width, height } = imageData;
  const points: Point[] = [];

  for (let y = sampleRate; y < height - sampleRate; y += sampleRate) {
    for (let x = sampleRate; x < width - sampleRate; x += sampleRate) {
      const idx = (y * width + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];

      const idxR = (y * width + Math.min(width - 1, x + sampleRate)) * 4;
      const idxD = (Math.min(height - 1, y + sampleRate) * width + x) * 4;
      const idxL = (y * width + Math.max(0, x - sampleRate)) * 4;
      const idxU = (Math.max(0, y - sampleRate) * width + x) * 4;

      const gradX = Math.abs(r - data[idxR]) + Math.abs(g - data[idxR + 1]) + Math.abs(b - data[idxR + 2]);
      const gradY = Math.abs(r - data[idxD]) + Math.abs(g - data[idxD + 1]) + Math.abs(b - data[idxD + 2]);
      const gradient = (gradX + gradY) / (3 * 255 * 2);

      const maxC = Math.max(r, g, b) / 255;
      const minC = Math.min(r, g, b) / 255;
      const saturation = maxC === 0 ? 0 : (maxC - minC) / maxC;
      const value = maxC;

      const localVar = (
        Math.abs(r - data[idxL]) + Math.abs(g - data[idxL + 1]) + Math.abs(b - data[idxL + 2]) +
        Math.abs(r - data[idxU]) + Math.abs(g - data[idxU + 1]) + Math.abs(b - data[idxU + 2])
      ) / (6 * 255);

      let score = 0;
      if (mode === 'artifacts') {
        score = gradient * 0.5 + Math.abs(saturation - 0.4) * 0.3 + localVar * 0.2;
      } else if (mode === 'noise') {
        score = localVar * 0.6 + gradient * 0.25 + Math.abs(value - 0.5) * 0.15;
      } else if (mode === 'lighting') {
        score = Math.abs(value - 0.5) * 0.5 + gradient * 0.25 + (1 - saturation) * value * 0.25;
      }

      if (score > 0.12) points.push({ x, y, value: Math.min(1, score) });
    }
  }
  return points;
}

function dbscan(points: Point[], epsilon: number, minPts: number): number[] {
  const n = points.length;
  const labels = new Array(n).fill(-1);
  let clusterId = 0;

  const regionQuery = (idx: number): number[] => {
    const p = points[idx];
    const neighbors: number[] = [];
    for (let i = 0; i < n; i++) {
      const q = points[i];
      if ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 <= epsilon ** 2) neighbors.push(i);
    }
    return neighbors;
  };

  const expandCluster = (idx: number, neighbors: number[], cId: number) => {
    labels[idx] = cId;
    const queue = [...neighbors];
    const visited = new Set(neighbors);
    let i = 0;
    while (i < queue.length) {
      const ni = queue[i];
      if (labels[ni] === -1) {
        labels[ni] = cId;
        const newN = regionQuery(ni);
        if (newN.length >= minPts) {
          newN.forEach(nn => { if (!visited.has(nn)) { visited.add(nn); queue.push(nn); } });
        }
      }
      i++;
    }
  };

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1) continue;
    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) { labels[i] = -1; continue; }
    clusterId++;
    expandCluster(i, neighbors, clusterId);
  }
  return labels;
}

function renderHeatmapToCanvas(canvas: HTMLCanvasElement, imageSrc: string, mode: HeatmapMode): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const W = img.naturalWidth;
      const H = img.naturalHeight;
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d')!;

      ctx.drawImage(img, 0, 0, W, H);
      const imageData = ctx.getImageData(0, 0, W, H);

      const sampleRate = Math.max(2, Math.floor(Math.min(W, H) / 120));
      const points = extractAnomalyPoints(imageData, mode, sampleRate);

      ctx.drawImage(img, 0, 0, W, H);

      if (points.length === 0) { resolve(); return; }

      const maxPts = 600;
      const sampled = points.length > maxPts
        ? points.filter((_, i) => i % Math.ceil(points.length / maxPts) === 0)
        : points;

      const epsilon = sampleRate * 7;
      const labels = dbscan(sampled, epsilon, 3);

      const clusterPts: Record<number, Point[]> = {};
      sampled.forEach((p, i) => {
        const l = labels[i];
        if (l > 0) { if (!clusterPts[l]) clusterPts[l] = []; clusterPts[l].push(p); }
      });

      const clusterSizes = Object.fromEntries(Object.entries(clusterPts).map(([k, v]) => [k, v.length]));
      const maxSize = Math.max(...Object.values(clusterSizes), 1);

      ctx.globalCompositeOperation = 'source-over';
      Object.entries(clusterPts).forEach(([cId, cPts]) => {
        const intensity = Math.min(1, clusterSizes[Number(cId)] / maxSize);
        const avgVal = cPts.reduce((s, p) => s + p.value, 0) / cPts.length;
        const combined = intensity * 0.6 + avgVal * 0.4;

        const cx = cPts.reduce((s, p) => s + p.x, 0) / cPts.length;
        const cy = cPts.reduce((s, p) => s + p.y, 0) / cPts.length;
        const spread = Math.sqrt(
          cPts.reduce((s, p) => s + (p.x - cx) ** 2 + (p.y - cy) ** 2, 0) / cPts.length
        );
        const radius = Math.max(epsilon * 1.5, spread * 2);

        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        if (combined > 0.55) {
          grad.addColorStop(0, `rgba(255, 20, 20, ${Math.min(0.85, combined * 0.9)})`);
          grad.addColorStop(0.35, `rgba(255, 80, 0, ${combined * 0.5})`);
          grad.addColorStop(0.7, `rgba(255, 150, 0, ${combined * 0.2})`);
          grad.addColorStop(1, 'rgba(255, 0, 0, 0)');
        } else if (combined > 0.3) {
          grad.addColorStop(0, `rgba(255, 210, 0, ${Math.min(0.75, combined * 0.85)})`);
          grad.addColorStop(0.4, `rgba(255, 160, 0, ${combined * 0.4})`);
          grad.addColorStop(1, 'rgba(255, 200, 0, 0)');
        } else {
          grad.addColorStop(0, `rgba(0, 220, 255, ${Math.min(0.6, combined * 0.8)})`);
          grad.addColorStop(0.5, `rgba(0, 120, 255, ${combined * 0.3})`);
          grad.addColorStop(1, 'rgba(0, 200, 255, 0)');
        }
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
      });

      sampled.forEach((p, i) => {
        if (labels[i] === -1 && p.value > 0.35) {
          ctx.fillStyle = `rgba(255, 255, 100, ${p.value * 0.25})`;
          ctx.fillRect(p.x - 1, p.y - 1, 3, 3);
        }
      });

      resolve();
    };
    img.onerror = reject;
    img.src = imageSrc;
  });
}

// ============================================================
// LOGIN SCREEN
// ============================================================
const LoginScreen = ({ onLogin }: { onLogin: () => void }) => {
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isSuccess, setIsSuccess] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { if (videoRef.current) videoRef.current.srcObject = stream; })
      .catch(err => console.error("Camera denied", err));
  }, []);

  useEffect(() => {
    if (scanning && progress < 100) {
      const iv = setInterval(() => setProgress(p => {
        const n = p + 1.5;
        if (n >= 100) { clearInterval(iv); setIsSuccess(true); setTimeout(onLogin, 800); return 100; }
        return n;
      }), 20);
      return () => clearInterval(iv);
    } else if (!scanning && progress < 100 && progress > 0) {
      const iv = setInterval(() => setProgress(p => {
        const n = p - 4; if (n <= 0) { clearInterval(iv); return 0; } return n;
      }), 20);
      return () => clearInterval(iv);
    }
  }, [scanning, progress, onLogin]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-guardian-bg overflow-hidden relative">
      <div className="absolute inset-0 opacity-10 pointer-events-none"
        style={{ backgroundImage: 'radial-gradient(circle, #00f2ff 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="z-10 text-center space-y-8 max-w-md w-full">
        <div className="flex justify-center">
          <div className="relative">
            <motion.div animate={{ scale: [1, 1.05, 1] }} transition={{ duration: 4, repeat: Infinity }}
              className="p-4 rounded-full bg-guardian-primary/10 border border-guardian-primary/30">
              <Shield className="w-16 h-16 text-guardian-primary" />
            </motion.div>
            <div className="absolute -inset-4 border border-guardian-primary/20 rounded-full animate-pulse" />
          </div>
        </div>
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tighter text-white uppercase">Guardian AI</h1>
          <p className="text-guardian-primary/60 font-mono text-sm uppercase tracking-widest">Neural Security Protocol v4.0</p>
        </div>
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-guardian-primary to-guardian-secondary rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
          <div className="relative bg-guardian-card border border-white/10 rounded-2xl p-8 space-y-6">
            <div className="relative aspect-square w-48 mx-auto rounded-full overflow-hidden border-2 border-guardian-primary/50 bg-black">
              <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 w-full h-full object-cover grayscale opacity-70" />
              <div className="absolute inset-0 border-2 border-guardian-primary/30 rounded-full animate-ping" />
              <div className="absolute top-1/2 left-0 w-full h-0.5 bg-guardian-primary/50 shadow-[0_0_10px_#00f2ff] animate-scan-vertical" />
            </div>
            <div className="space-y-4">
              <p className="text-xs font-mono text-guardian-primary/80 uppercase tracking-tight">
                {isSuccess ? "Authentication Successful" : "Biometric Verification Required"}
              </p>
              <div className="relative w-24 h-24 mx-auto">
                <motion.button
                  onMouseDown={() => { if (!isSuccess) setScanning(true); }}
                  onMouseUp={() => setScanning(false)}
                  onMouseLeave={() => setScanning(false)}
                  onTouchStart={() => { if (!isSuccess) setScanning(true); }}
                  onTouchEnd={() => setScanning(false)}
                  animate={{ scale: scanning ? 0.95 : 1, borderColor: isSuccess ? "#22c55e" : (scanning ? "#00f2ff" : "rgba(255,255,255,0.1)") }}
                  className={cn("relative w-full h-full flex items-center justify-center rounded-2xl border-2 transition-colors duration-300 bg-white/5 overflow-hidden",
                    isSuccess && "bg-green-500/10")}>
                  <Fingerprint className={cn("w-12 h-12 transition-colors duration-300",
                    isSuccess ? "text-green-500" : (scanning ? "text-guardian-primary" : "text-white/40"))} />
                  {scanning && !isSuccess && (
                    <motion.div initial={{ top: "0%" }} animate={{ top: "100%" }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      className="absolute left-0 w-full h-0.5 bg-guardian-primary shadow-[0_0_10px_#00f2ff] z-10" />
                  )}
                  <motion.div initial={{ height: 0 }} animate={{ height: `${progress}%` }}
                    transition={{ type: "spring", bounce: 0, duration: 0.2 }}
                    className={cn("absolute bottom-0 left-0 w-full pointer-events-none",
                      isSuccess ? "bg-green-500/20" : "bg-guardian-primary/20")} />
                </motion.button>
              </div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden max-w-[120px] mx-auto">
                <motion.div className={cn("h-full", isSuccess ? "bg-green-500" : "bg-guardian-primary")}
                  animate={{ width: `${progress}%` }} transition={{ type: "spring", bounce: 0, duration: 0.2 }} />
              </div>
              <p className="text-[10px] font-mono text-white/30 uppercase">
                {isSuccess ? "Access Granted" : (scanning ? `Scanning... ${Math.round(progress)}%` : "Hold to authenticate")}
              </p>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-center gap-4 text-[10px] font-mono text-white/20 uppercase tracking-widest">
          <span className="flex items-center gap-1"><Lock className="w-3 h-3" /> Encrypted</span>
          <span className="flex items-center gap-1"><Cpu className="w-3 h-3" /> Neural Link</span>
        </div>
      </motion.div>
    </div>
  );
};

// ============================================================
// DBSCAN HEATMAP PANEL
// ============================================================
const DBSCANPanel = ({ image }: { image: string }) => {
  const [activeMode, setActiveMode] = useState<HeatmapMode>('artifacts');
  const [generating, setGenerating] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasUrl, setCanvasUrl] = useState<string | null>(null);

  const runHeatmap = useCallback(async (mode: HeatmapMode) => {
    if (!canvasRef.current) return;
    setGenerating(true);
    setCanvasUrl(null);
    try {
      await renderHeatmapToCanvas(canvasRef.current, image, mode);
      setCanvasUrl(canvasRef.current.toDataURL());
    } catch (e) {
      console.error('DBSCAN failed', e);
    }
    setGenerating(false);
  }, [image]);

  useEffect(() => { runHeatmap(activeMode); }, [activeMode, runHeatmap]);

  const handleMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX : e.clientX;
    setSliderPos(Math.min(Math.max(((x - rect.left) / rect.width) * 100, 0), 100));
  };

  return (
    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
      className="bg-guardian-card border border-guardian-primary/20 rounded-2xl overflow-hidden">

      {/* Header */}
      <div className="p-5 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-guardian-primary/10 border border-guardian-primary/20">
            <Activity className="w-5 h-5 text-guardian-primary" />
          </div>
          <div>
            <p className="text-sm font-bold uppercase tracking-widest text-white">DBSCAN Neural Heatmap</p>
            <p className="text-[10px] font-mono text-white/40">Density-based spatial anomaly clustering • Client-side analysis</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {generating && <RefreshCw className="w-4 h-4 text-guardian-primary animate-spin" />}
          <span className={cn("text-[9px] font-mono uppercase px-2 py-1 rounded",
            generating ? "text-guardian-primary bg-guardian-primary/10" : "text-green-400 bg-green-400/10")}>
            {generating ? "Analyzing..." : "Ready"}
          </span>
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* Mode Tabs */}
        <div className="grid grid-cols-3 gap-2">
          {(['artifacts', 'noise', 'lighting'] as HeatmapMode[]).map((mode) => (
            <button key={mode} onClick={() => setActiveMode(mode)}
              className={cn(
                "py-3 rounded-xl text-[10px] font-mono uppercase tracking-widest transition-all border",
                activeMode === mode
                  ? "bg-guardian-primary/10 border-guardian-primary text-guardian-primary shadow-[0_0_15px_rgba(0,242,255,0.15)]"
                  : "bg-white/5 border-transparent text-white/40 hover:text-white/60 hover:bg-white/10"
              )}>
              <div className="space-y-0.5">
                <div>{mode}</div>
                <div className="text-[8px] opacity-60">
                  {mode === 'artifacts' ? 'Edge anomalies' : mode === 'noise' ? 'Pixel variance' : 'Value inconsistency'}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Main Heatmap Viewer */}
        <div ref={containerRef} onMouseMove={handleMove} onTouchMove={handleMove}
          className="relative w-full rounded-xl overflow-hidden bg-black cursor-ew-resize select-none"
          style={{ aspectRatio: '16/9' }}>

          {/* Original base */}
          <img src={image} className="absolute inset-0 w-full h-full object-contain" alt="Original" />

          {/* Heatmap overlay */}
          {canvasUrl && !generating && (
            <div className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}>
              <img src={canvasUrl} className="absolute inset-0 w-full h-full object-contain" alt="DBSCAN Heatmap" />
              <div className="absolute top-3 left-3 bg-black/80 border border-guardian-primary/40 text-guardian-primary text-[9px] font-mono px-2 py-1 rounded-lg uppercase tracking-widest animate-pulse">
                ⬡ DBSCAN {activeMode}
              </div>
            </div>
          )}

          {/* Generating overlay */}
          {generating && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/75 z-20">
              <div className="text-center space-y-3 p-6">
                <div className="relative mx-auto w-12 h-12">
                  <RefreshCw className="w-12 h-12 text-guardian-primary animate-spin" />
                </div>
                <p className="text-[11px] font-mono text-guardian-primary uppercase tracking-widest">Running DBSCAN Analysis</p>
                <p className="text-[9px] font-mono text-white/30">Clustering spatial anomaly regions...</p>
              </div>
            </div>
          )}

          {/* Slider divider */}
          {canvasUrl && !generating && (
            <div className="absolute top-0 bottom-0 z-30 pointer-events-none"
              style={{ left: `${sliderPos}%` }}>
              <div className="absolute inset-0 w-px bg-guardian-primary shadow-[0_0_12px_#00f2ff]" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-guardian-primary border-2 border-black flex items-center justify-center shadow-lg">
                <ZoomIn className="w-3.5 h-3.5 text-black" />
              </div>
            </div>
          )}

          {/* Corner labels */}
          {canvasUrl && !generating && (
            <>
              <div className="absolute bottom-3 left-3 pointer-events-none z-10">
                <span className="text-[9px] font-mono bg-black/70 px-2 py-1 rounded text-guardian-primary uppercase border border-guardian-primary/20">← Heatmap</span>
              </div>
              <div className="absolute bottom-3 right-3 pointer-events-none z-10">
                <span className="text-[9px] font-mono bg-black/70 px-2 py-1 rounded text-white/50 uppercase">Original →</span>
              </div>
            </>
          )}
        </div>

        {/* Legend */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { color: 'bg-red-500', shadow: 'shadow-red-500/30', label: 'High Risk', desc: 'Dense anomaly cluster' },
            { color: 'bg-yellow-400', shadow: 'shadow-yellow-400/30', label: 'Medium Risk', desc: 'Moderate anomalies' },
            { color: 'bg-cyan-400', shadow: 'shadow-cyan-400/30', label: 'Low Risk', desc: 'Minor irregularities' },
          ].map((item, i) => (
            <div key={i} className="bg-white/[0.03] border border-white/5 p-3 rounded-xl flex items-start gap-2">
              <div className={cn("w-3 h-3 rounded mt-0.5 shrink-0 shadow-md", item.color, item.shadow)} />
              <div>
                <p className="text-[10px] font-mono text-white/70 uppercase font-bold">{item.label}</p>
                <p className="text-[8px] font-mono text-white/30 mt-0.5">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <p className="text-[9px] font-mono text-white/20 text-center border-t border-white/5 pt-3">
          Drag slider ← → to compare DBSCAN heatmap with original image • Switch modes to analyze different anomaly types
        </p>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </motion.div>
  );
};

// ============================================================
// DASHBOARD
// ============================================================
const Dashboard = () => {
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    multiple: false
  } as any);

  const handleAnalyze = async () => {
    if (!image) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyzeImage(image);
      setResult(res);
    } catch (err: any) {
      const isQuota = err?.message?.includes("RESOURCE_EXHAUSTED") || err?.code === 429;
      setError(isQuota ? "System quota exceeded. Please wait and try again." : "Analysis failed. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-guardian-bg p-6 md:p-10">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Shield className="w-6 h-6 text-guardian-primary" />
              <h1 className="text-2xl font-bold tracking-tight uppercase text-white">Guardian Terminal</h1>
            </div>
            <p className="text-white/40 text-sm font-mono">Session ID: {Math.random().toString(36).substring(7).toUpperCase()}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right hidden md:block">
              <p className="text-[10px] font-mono text-white/30 uppercase">System Status</p>
              <p className="text-xs font-mono text-guardian-primary flex items-center gap-1 justify-end">
                <span className="w-2 h-2 rounded-full bg-guardian-primary animate-pulse" /> Operational
              </p>
            </div>
            <button onClick={() => window.location.reload()}
              className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
              <RefreshCw className="w-5 h-5 text-white/60" />
            </button>
          </div>
        </header>

        {/* Upload + Results */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Upload */}
          <div className="lg:col-span-7 space-y-4">
            <div {...getRootProps()}
              className={cn(
                "relative aspect-video rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-8 cursor-pointer overflow-hidden group/container",
                isDragActive ? "border-guardian-primary bg-guardian-primary/5" : "border-white/10 bg-guardian-card hover:border-white/20"
              )}>
              <input {...getInputProps()} />
              {image ? (
                <div className="absolute inset-0 w-full h-full">
                  <img src={image} className="absolute inset-0 w-full h-full object-contain" alt="Uploaded" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/container:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                    <p className="text-sm font-mono uppercase tracking-widest bg-black/60 px-4 py-2 rounded-full text-white">Change Image</p>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-full bg-white/5 flex items-center justify-center">
                    <Upload className="w-8 h-8 text-white/40" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-lg font-medium text-white">Drop image for analysis</p>
                    <p className="text-sm text-white/40 font-mono">JPG, PNG, WEBP up to 10MB</p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-4">
              <button onClick={handleAnalyze} disabled={!image || analyzing}
                className={cn(
                  "flex-1 py-4 rounded-xl font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2",
                  !image || analyzing ? "bg-white/5 text-white/20 cursor-not-allowed" : "bg-guardian-primary text-black hover:shadow-[0_0_20px_#00f2ff] active:scale-95"
                )}>
                {analyzing ? <><RefreshCw className="w-5 h-5 animate-spin" /> Analyzing...</> : <><Search className="w-5 h-5" /> Execute Deep Scan</>}
              </button>
              {image && !analyzing && (
                <button onClick={() => { setImage(null); setResult(null); setError(null); }}
                  className="px-6 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-white">
                  Clear
                </button>
              )}
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { icon: Cpu, label: "Neural Engine", value: "Active" },
                { icon: Activity, label: "Algorithm", value: "DBSCAN" },
                { icon: Lock, label: "Encryption", value: "AES-256" }
              ].map((t, i) => (
                <div key={i} className="bg-guardian-card border border-white/5 p-3 rounded-xl space-y-1">
                  <t.icon className="w-4 h-4 text-guardian-primary/60" />
                  <p className="text-[10px] font-mono text-white/30 uppercase">{t.label}</p>
                  <p className="text-xs font-mono text-white/80">{t.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-5 space-y-4">
            <AnimatePresence mode="wait">
              {!result && !analyzing && !error && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="h-full flex flex-col items-center justify-center text-center p-12 border border-white/5 rounded-2xl bg-white/[0.02] min-h-[300px]">
                  <Info className="w-12 h-12 text-white/10 mb-4" />
                  <h3 className="text-lg font-medium text-white/40">Awaiting Input</h3>
                  <p className="text-sm text-white/20 max-w-xs mt-2">Upload an image and execute deep scan to begin.</p>
                </motion.div>
              )}

              {analyzing && (
                <motion.div key="analyzing" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="bg-guardian-card border border-guardian-primary/20 p-8 rounded-2xl space-y-6 min-h-[300px]">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-mono text-guardian-primary uppercase tracking-widest">Processing...</h3>
                    <span className="text-xs font-mono text-white/40">AI Analysis</span>
                  </div>
                  <div className="space-y-4">
                    {["Isolating pixel artifacts", "Running DBSCAN clustering", "Checking metadata signatures", "Neural pattern matching"].map((step, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <div className={cn("w-1.5 h-1.5 rounded-full", i === 1 ? "bg-guardian-primary animate-pulse" : "bg-white/10")} />
                        <p className={cn("text-xs font-mono", i === 1 ? "text-white" : "text-white/30")}>{step}</p>
                      </div>
                    ))}
                  </div>
                  <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                    <motion.div className="h-full bg-guardian-primary"
                      animate={{ x: ["-100%", "100%"] }} transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }} />
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div key="error" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="bg-red-500/10 border border-red-500/20 p-6 rounded-2xl flex items-start gap-4">
                  <AlertTriangle className="w-6 h-6 text-red-500 shrink-0" />
                  <div>
                    <p className="font-bold text-red-500 uppercase text-sm">System Error</p>
                    <p className="text-sm text-red-500/80">{error}</p>
                  </div>
                </motion.div>
              )}

              {result && (
                <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  <div className={cn("p-6 rounded-2xl border-2 space-y-5 relative overflow-hidden",
                    result.isReal ? "border-green-500/30 bg-green-500/5" : "border-red-500/30 bg-red-500/5")}>
                    <div className="flex items-center justify-between relative z-10">
                      <div className="space-y-1">
                        <p className="text-[10px] font-mono uppercase tracking-widest text-white/40">Final Verdict</p>
                        <h2 className={cn("text-3xl font-black uppercase tracking-tighter",
                          result.isReal ? "text-green-500" : "text-red-500")}>
                          {result.isReal ? "Authentic" : "Synthetic"}
                        </h2>
                      </div>
                      {result.isReal ? <CheckCircle2 className="w-12 h-12 text-green-500" /> : <AlertTriangle className="w-12 h-12 text-red-500" />}
                    </div>
                    <div className="grid grid-cols-2 gap-3 relative z-10">
                      <div className="bg-black/20 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] font-mono text-white/30 uppercase mb-1">Confidence</p>
                        <p className="text-xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-black/20 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] font-mono text-white/30 uppercase mb-1">Source</p>
                        <p className="text-xl font-bold truncate text-white">{result.source}</p>
                      </div>
                    </div>
                    <div className="space-y-1 relative z-10">
                      <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest">Analysis Summary</p>
                      <p className="text-sm text-white/70 leading-relaxed italic">"{result.reasoning}"</p>
                    </div>
                  </div>

                  {result.metadata.artifacts && result.metadata.artifacts.length > 0 && (
                    <div className="bg-guardian-card border border-white/5 p-5 rounded-2xl space-y-3">
                      <h4 className="text-xs font-mono text-white/40 uppercase tracking-widest">Detected Anomalies</h4>
                      <div className="space-y-2">
                        {result.metadata.artifacts.map((artifact, i) => (
                          <div key={i} className="flex items-center gap-3 p-2.5 rounded-lg bg-white/[0.02] border border-white/5">
                            <div className="w-1.5 h-1.5 rounded-full bg-guardian-primary shrink-0" />
                            <p className="text-xs text-white/60">{artifact}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* DBSCAN Heatmap Section — auto appears after scan */}
        {result && image && <DBSCANPanel image={image} />}

      </div>
    </div>
  );
};

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  return (
    <div className="min-h-screen font-sans bg-guardian-bg text-white">
      <AnimatePresence mode="wait">
        {!isLoggedIn ? (
          <motion.div key="login" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <LoginScreen onLogin={() => setIsLoggedIn(true)} />
          </motion.div>
        ) : (
          <motion.div key="dashboard" initial={{ opacity: 0, scale: 1.05 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
            <Dashboard />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}