import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Shield, 
  Fingerprint, 
  Camera, 
  Upload, 
  Search, 
  AlertTriangle, 
  CheckCircle2, 
  Info, 
  Cpu, 
  Activity,
  Lock,
  RefreshCw
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { cn } from './lib/utils';
import { analyzeImage, generateHeatmap, AnalysisResult, HeatmapMode } from './services/gemini';

// --- Components ---

const LoginScreen = ({ onLogin }: { onLogin: () => void }) => {
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isSuccess, setIsSuccess] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera access denied", err);
      }
    }
    setupCamera();
  }, []);

  useEffect(() => {
    if (scanning && progress < 100) {
      const interval = setInterval(() => {
        setProgress(prev => {
          const next = prev + 1.5;
          if (next >= 100) {
            clearInterval(interval);
            setIsSuccess(true);
            setTimeout(onLogin, 800);
            return 100;
          }
          return next;
        });
      }, 20);
      return () => clearInterval(interval);
    } else if (!scanning && progress < 100) {
      // Reset progress if user lets go before 100%
      const interval = setInterval(() => {
        setProgress(prev => {
          const next = prev - 4;
          if (next <= 0) {
            clearInterval(interval);
            return 0;
          }
          return next;
        });
      }, 20);
      return () => clearInterval(interval);
    }
  }, [scanning, progress, onLogin]);

  const handleStartScan = () => {
    if (isSuccess) return;
    setScanning(true);
  };

  const handleEndScan = () => {
    setScanning(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-guardian-bg overflow-hidden relative">
      {/* Background Grid */}
      <div className="absolute inset-0 opacity-10 pointer-events-none" 
           style={{ backgroundImage: 'radial-gradient(circle, #00f2ff 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="z-10 text-center space-y-8 max-w-md w-full"
      >
        <div className="flex justify-center">
          <div className="relative">
            <motion.div 
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 4, repeat: Infinity }}
              className="p-4 rounded-full bg-guardian-primary/10 border border-guardian-primary/30"
            >
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
          <div className="absolute -inset-1 bg-gradient-to-r from-guardian-primary to-guardian-secondary rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative bg-guardian-card border border-white/10 rounded-2xl p-8 space-y-6">
            
            {/* Camera Feed Simulation */}
            <div className="relative aspect-square w-48 mx-auto rounded-full overflow-hidden border-2 border-guardian-primary/50 bg-black">
              <video 
                ref={videoRef} 
                autoPlay 
                muted 
                playsInline 
                className="absolute inset-0 w-full h-full object-cover grayscale opacity-70"
              />
              <div className="absolute inset-0 border-2 border-guardian-primary/30 rounded-full animate-ping" />
              <div className="absolute top-1/2 left-0 w-full h-0.5 bg-guardian-primary/50 shadow-[0_0_10px_#00f2ff] animate-scan-vertical" />
            </div>

            <div className="space-y-4">
              <p className="text-xs font-mono text-guardian-primary/80 uppercase tracking-tight">
                {isSuccess ? "Authentication Successful" : "Biometric Verification Required"}
              </p>
              
              <div className="relative w-24 h-24 mx-auto">
                <motion.button 
                  onMouseDown={handleStartScan}
                  onMouseUp={handleEndScan}
                  onMouseLeave={handleEndScan}
                  onTouchStart={handleStartScan}
                  onTouchEnd={handleEndScan}
                  animate={{ 
                    scale: scanning ? 0.95 : 1,
                    borderColor: isSuccess ? "#22c55e" : (scanning ? "#00f2ff" : "rgba(255,255,255,0.1)")
                  }}
                  className={cn(
                    "relative w-full h-full flex items-center justify-center rounded-2xl border-2 transition-colors duration-300 bg-white/5 overflow-hidden",
                    isSuccess && "bg-green-500/10 shadow-[0_0_20px_rgba(34,197,94,0.4)]"
                  )}
                >
                  <Fingerprint className={cn(
                    "w-12 h-12 transition-colors duration-300", 
                    isSuccess ? "text-green-500" : (scanning ? "text-guardian-primary" : "text-white/40")
                  )} />
                  
                  {/* Scanning Bar on Icon */}
                  {scanning && !isSuccess && (
                    <motion.div 
                      initial={{ top: "0%" }}
                      animate={{ top: "100%" }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      className="absolute left-0 w-full h-0.5 bg-guardian-primary shadow-[0_0_10px_#00f2ff] z-10"
                    />
                  )}

                  {/* Progress Fill */}
                  <motion.div 
                    initial={{ height: 0 }}
                    animate={{ height: `${progress}%` }}
                    transition={{ type: "spring", bounce: 0, duration: 0.2 }}
                    className={cn(
                      "absolute bottom-0 left-0 w-full pointer-events-none",
                      isSuccess ? "bg-green-500/20" : "bg-guardian-primary/20"
                    )}
                  />
                </motion.button>

                {/* Success Glow Ring */}
                {isSuccess && (
                  <motion.div 
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1.5, opacity: 0 }}
                    transition={{ duration: 0.8 }}
                    className="absolute inset-0 border-2 border-green-500 rounded-2xl pointer-events-none"
                  />
                )}
              </div>
              
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden max-w-[120px] mx-auto">
                <motion.div 
                  className={cn(
                    "h-full shadow-[0_0_8px]",
                    isSuccess ? "bg-green-500 shadow-green-500/50" : "bg-guardian-primary shadow-guardian-primary/50"
                  )}
                  animate={{ width: `${progress}%` }}
                  transition={{ type: "spring", bounce: 0, duration: 0.2 }}
                />
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

const Dashboard = () => {
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [activeMode, setActiveMode] = useState<HeatmapMode>('artifacts');
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!containerRef.current || !showHeatmap) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const position = ((x - rect.left) / rect.width) * 100;
    setSliderPos(Math.min(Math.max(position, 0), 100));
  };

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
    accept: { 
      'image/jpeg': [], 
      'image/png': [], 
      'image/webp': [] 
    },
    multiple: false 
  } as any);

  const handleAnalyze = async () => {
    if (!image) return;
    setAnalyzing(true);
    setError(null);
    setShowHeatmap(false);
    try {
      const res = await analyzeImage(image);
      
      // If it's fake, generate advanced heatmaps using nano banana (gemini-2.5-flash-image)
      if (!res.isReal) {
        res.heatmaps = {};
        const modes: HeatmapMode[] = ['artifacts', 'noise', 'lighting'];
        
        await Promise.all(modes.map(async (mode) => {
          try {
            const url = await generateHeatmap(image, mode);
            if (res.heatmaps) res.heatmaps[mode] = url;
          } catch (hErr) {
            console.warn(`${mode} heatmap failed`, hErr);
          }
        }));
      }
      
      setResult(res);
    } catch (err: any) {
      const isQuotaError = err?.message?.includes("RESOURCE_EXHAUSTED") || err?.status === "RESOURCE_EXHAUSTED" || err?.code === 429;
      if (isQuotaError) {
        setError("System quota exceeded. Please wait a moment and try again.");
      } else {
        setError("Analysis failed. Please try again.");
      }
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-guardian-bg p-6 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-8">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Shield className="w-6 h-6 text-guardian-primary" />
              <h1 className="text-2xl font-bold tracking-tight uppercase text-white">Guardian Terminal</h1>
            </div>
            <p className="text-white/40 text-sm font-mono">Session ID: {Math.random().toString(36).substring(7).toUpperCase()}</p>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right hidden md:block">
              <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest">System Status</p>
              <p className="text-xs font-mono text-guardian-primary flex items-center gap-1 justify-end">
                <span className="w-2 h-2 rounded-full bg-guardian-primary animate-pulse" />
                Operational
              </p>
            </div>
            <button 
              onClick={() => window.location.reload()}
              className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors"
            >
              <RefreshCw className="w-5 h-5 text-white/60" />
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Upload & Preview */}
          <div className="lg:col-span-7 space-y-6">
            <div 
              {...getRootProps()} 
              ref={containerRef}
              onMouseMove={handleMouseMove}
              onTouchMove={handleMouseMove}
              className={cn(
                "relative aspect-video rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-8 cursor-pointer overflow-hidden group/container",
                isDragActive ? "border-guardian-primary bg-guardian-primary/5" : "border-white/10 bg-guardian-card hover:border-white/20"
              )}
            >
              <input {...getInputProps()} />
              
              {image ? (
                <div className="absolute inset-0 w-full h-full select-none">
                  {/* Original Image (Background) */}
                  <img 
                    src={image} 
                    className="absolute inset-0 w-full h-full object-contain" 
                    alt="Original" 
                    referrerPolicy="no-referrer" 
                  />

                  {/* Heatmap Overlay (Clipped) */}
                  {showHeatmap && result?.heatmaps && result.heatmaps[activeMode] && (
                    <>
                      <div 
                        className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
                        style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                      >
                        <img 
                          src={result.heatmaps[activeMode]} 
                          className="absolute inset-0 w-full h-full object-contain" 
                          alt="Heatmap" 
                          referrerPolicy="no-referrer" 
                        />
                        <div className="absolute top-4 left-4 bg-red-500/80 text-white text-[10px] font-mono px-2 py-1 rounded uppercase tracking-widest animate-pulse z-30">
                          Neural {activeMode} Analysis
                        </div>
                      </div>

                      {/* Slider Bar */}
                      <div 
                        className="absolute top-0 bottom-0 w-0.5 bg-guardian-primary shadow-[0_0_15px_#00f2ff] z-40 cursor-ew-resize pointer-events-none"
                        style={{ left: `${sliderPos}%` }}
                      >
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-guardian-primary border-4 border-guardian-bg flex items-center justify-center shadow-lg">
                          <RefreshCw className="w-4 h-4 text-black animate-spin-slow" />
                        </div>
                      </div>

                      {/* Labels */}
                      <div className="absolute bottom-4 left-4 z-30 pointer-events-none">
                        <span className="text-[10px] font-mono bg-black/60 px-2 py-1 rounded text-white/60 uppercase tracking-tighter">Heatmap</span>
                      </div>
                      <div className="absolute bottom-4 right-4 z-30 pointer-events-none">
                        <span className="text-[10px] font-mono bg-black/60 px-2 py-1 rounded text-white/60 uppercase tracking-tighter">Original</span>
                      </div>

                      {/* Legend */}
                      <div className="absolute top-4 right-4 bg-black/80 border border-white/10 p-2 rounded-lg space-y-1.5 z-50 backdrop-blur-sm pointer-events-none">
                        <p className="text-[9px] font-mono text-white/40 uppercase tracking-widest border-b border-white/5 pb-1 mb-1">Neural Legend</p>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-sm bg-red-500 shadow-[0_0_5px_#ef4444]" />
                          <span className="text-[9px] font-mono text-white/80">High Risk</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-sm bg-yellow-500 shadow-[0_0_5px_#eab308]" />
                          <span className="text-[9px] font-mono text-white/80">Moderate</span>
                        </div>
                      </div>
                    </>
                  )}

                  {!showHeatmap && (
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/container:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                      <p className="text-sm font-mono uppercase tracking-widest bg-black/60 px-4 py-2 rounded-full text-white">Change Image</p>
                    </div>
                  )}
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
              <button 
                onClick={handleAnalyze}
                disabled={!image || analyzing}
                className={cn(
                  "flex-1 py-4 rounded-xl font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2",
                  !image || analyzing 
                    ? "bg-white/5 text-white/20 cursor-not-allowed" 
                    : "bg-guardian-primary text-black hover:shadow-[0_0_20px_#00f2ff] active:scale-95"
                )}
              >
                {analyzing ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Analyzing Neural Patterns...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Execute Deep Scan
                  </>
                )}
              </button>
              {image && !analyzing && (
                <button 
                  onClick={() => { setImage(null); setResult(null); setShowHeatmap(false); }}
                  className="px-6 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-white"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Heatmap Toggle & Mode Selector */}
            {result && !result.isReal && result.heatmaps && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-guardian-card border border-white/5 p-4 rounded-xl space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-guardian-primary/10 border border-guardian-primary/20">
                      <Activity className="w-5 h-5 text-guardian-primary" />
                    </div>
                    <div>
                      <p className="text-xs font-bold uppercase tracking-tight text-white">Advanced Neural Analysis</p>
                      <p className="text-[10px] font-mono text-white/40">Multi-layer deepfake localization</p>
                    </div>
                  </div>
                  <button 
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className={cn(
                      "px-4 py-2 rounded-lg text-[10px] font-mono uppercase tracking-widest transition-all",
                      showHeatmap 
                        ? "bg-guardian-primary text-black shadow-[0_0_10px_rgba(0,242,255,0.5)]" 
                        : "bg-white/5 text-white/60 hover:bg-white/10"
                    )}
                  >
                    {showHeatmap ? "Deactivate Scanner" : "Activate Scanner"}
                  </button>
                </div>

                {showHeatmap && (
                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5">
                    {(['artifacts', 'noise', 'lighting'] as HeatmapMode[]).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setActiveMode(mode)}
                        className={cn(
                          "py-2 rounded-lg text-[9px] font-mono uppercase tracking-widest transition-all border",
                          activeMode === mode 
                            ? "bg-guardian-primary/10 border-guardian-primary text-guardian-primary" 
                            : "bg-white/5 border-transparent text-white/40 hover:text-white/60"
                        )}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                )}
              </motion.div>
            )}

            {/* Mock Tools */}
            <div className="grid grid-cols-3 gap-4">
              {[
                { icon: Cpu, label: "Neural Engine", value: "Active" },
                { icon: Activity, label: "Latency", value: "24ms" },
                { icon: Lock, label: "Encryption", value: "AES-256" }
              ].map((tool, i) => (
                <div key={i} className="bg-guardian-card border border-white/5 p-4 rounded-xl space-y-2">
                  <tool.icon className="w-4 h-4 text-guardian-primary/60" />
                  <p className="text-[10px] font-mono text-white/30 uppercase">{tool.label}</p>
                  <p className="text-xs font-mono text-white/80">{tool.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-5 space-y-6">
            <AnimatePresence mode="wait">
              {!result && !analyzing && !error && (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full flex flex-col items-center justify-center text-center p-12 border border-white/5 rounded-2xl bg-white/[0.02]"
                >
                  <Info className="w-12 h-12 text-white/10 mb-4" />
                  <h3 className="text-lg font-medium text-white/40">Awaiting Input</h3>
                  <p className="text-sm text-white/20 max-w-xs mt-2">Upload an image to begin the deepfake detection process.</p>
                </motion.div>
              )}

              {analyzing && (
                <motion.div 
                  key="analyzing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-6"
                >
                  <div className="bg-guardian-card border border-guardian-primary/20 p-8 rounded-2xl space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-mono text-guardian-primary uppercase tracking-widest">Processing...</h3>
                      <span className="text-xs font-mono text-white/40">Step 2/4</span>
                    </div>
                    <div className="space-y-4">
                      {[
                        "Isolating pixel artifacts",
                        "Analyzing lighting consistency",
                        "Checking metadata signatures",
                        "Neural pattern matching"
                      ].map((step, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <div className={cn(
                            "w-1.5 h-1.5 rounded-full",
                            i === 1 ? "bg-guardian-primary animate-pulse" : "bg-white/10"
                          )} />
                          <p className={cn(
                            "text-xs font-mono",
                            i === 1 ? "text-white" : "text-white/30"
                          )}>{step}</p>
                        </div>
                      ))}
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-guardian-primary"
                        animate={{ x: ["-100%", "100%"] }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div 
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-red-500/10 border border-red-500/20 p-6 rounded-2xl flex items-start gap-4"
                >
                  <AlertTriangle className="w-6 h-6 text-red-500 shrink-0" />
                  <div className="space-y-1">
                    <p className="font-bold text-red-500 uppercase text-sm">System Error</p>
                    <p className="text-sm text-red-500/80">{error}</p>
                  </div>
                </motion.div>
              )}

              {result && (
                <motion.div 
                  key="result"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-6"
                >
                  {/* Verdict Card */}
                  <div className={cn(
                    "p-8 rounded-2xl border-2 space-y-6 relative overflow-hidden",
                    result.isReal ? "border-green-500/30 bg-green-500/5" : "border-red-500/30 bg-red-500/5"
                  )}>
                    <div className="flex items-center justify-between relative z-10">
                      <div className="space-y-1">
                        <p className="text-[10px] font-mono uppercase tracking-widest text-white/40">Final Verdict</p>
                        <h2 className={cn(
                          "text-3xl font-black uppercase tracking-tighter",
                          result.isReal ? "text-green-500" : "text-red-500"
                        )}>
                          {result.isReal ? "Authentic" : "Synthetic"}
                        </h2>
                      </div>
                      {result.isReal ? (
                        <CheckCircle2 className="w-12 h-12 text-green-500" />
                      ) : (
                        <AlertTriangle className="w-12 h-12 text-red-500" />
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-4 relative z-10">
                      <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                        <p className="text-[10px] font-mono text-white/30 uppercase mb-1">Confidence</p>
                        <p className="text-xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                        <p className="text-[10px] font-mono text-white/30 uppercase mb-1">Source Origin</p>
                        <p className="text-xl font-bold truncate text-white">{result.source}</p>
                      </div>
                    </div>

                    <div className="space-y-2 relative z-10">
                      <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest">Analysis Summary</p>
                      <p className="text-sm text-white/70 leading-relaxed italic">
                        "{result.reasoning}"
                      </p>
                    </div>

                    {/* Background Icon */}
                    <div className="absolute -bottom-8 -right-8 opacity-5">
                      {result.isReal ? <CheckCircle2 className="w-48 h-48" /> : <AlertTriangle className="w-48 h-48" />}
                    </div>
                  </div>

                  {/* Artifacts List */}
                  {result.metadata.artifacts && result.metadata.artifacts.length > 0 && (
                    <div className="bg-guardian-card border border-white/5 p-6 rounded-2xl space-y-4">
                      <h4 className="text-xs font-mono text-white/40 uppercase tracking-widest">Detected Anomalies</h4>
                      <div className="space-y-2">
                        {result.metadata.artifacts.map((artifact, i) => (
                          <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.02] border border-white/5">
                            <div className="w-1.5 h-1.5 rounded-full bg-guardian-primary" />
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
          <motion.div 
            key="login"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <LoginScreen onLogin={() => setIsLoggedIn(true)} />
          </motion.div>
        ) : (
          <motion.div 
            key="dashboard"
            initial={{ opacity: 0, scale: 1.05 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Dashboard />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
