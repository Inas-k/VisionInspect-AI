import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  Cpu,
  Database,
  Gauge,
  History,
  ImageUp,
  Layers,
  MonitorDot,
  Play,
  RotateCcw,
  Search,
  Settings,
  ShieldCheck,
  Upload,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { API_BASE, getHealth, getHistory, getMetrics, getSamples, inspectImage, inspectSample, mediaUrl } from "./services/api";
import type { HistoryRecord, InspectionParameters, InspectionResult, SampleImage, StageKey } from "./types/api";

const defaultParams: InspectionParameters = {
  canny_low: 50,
  canny_high: 150,
  min_defect_area: 50,
  pass_fail_threshold: 1.0,
  blur_kernel: 5,
};

const navItems = [
  { id: "inspection", label: "INSPECTION", icon: Search },
  { id: "pipeline", label: "PIPELINE", icon: Layers },
  { id: "analytics", label: "ANALYTICS", icon: BarChart3 },
  { id: "history", label: "HISTORY", icon: History },
  { id: "system", label: "SYSTEM", icon: Cpu },
] as const;

type Page = (typeof navItems)[number]["id"];

const stages: { key: StageKey; label: string }[] = [
  { key: "original", label: "ORIGINAL" },
  { key: "grayscale", label: "GRAYSCALE" },
  { key: "edges", label: "EDGE MAP" },
  { key: "annotated", label: "DEFECT OVERLAY" },
];

function PageFrame({ children }: { children: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }} transition={{ duration: 0.18 }}>
      {children}
    </motion.div>
  );
}

function SystemBar({ online }: { online: boolean }) {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = window.setInterval(() => setTime(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <header className="topbar">
      <div>
        <h1>VISIONINSPECT AI</h1>
        <p>AUTOMATED INDUSTRIAL DEFECT DETECTION</p>
      </div>
      <div className="topbar-right">
        <span className={`online ${online ? "is-online" : "is-offline"}`} />
        <span>{online ? "VISION ENGINE ONLINE" : "VISION ENGINE OFFLINE"}</span>
        <span className="mono">{time.toLocaleTimeString()}</span>
        <span className="mode">MANUAL INSPECTION</span>
      </div>
    </header>
  );
}

function Sidebar({ page, setPage }: { page: Page; setPage: (page: Page) => void }) {
  return (
    <aside className="sidebar">
      <div className="mark"><MonitorDot size={24} /><span>VI</span></div>
      <nav>
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = item.id === page;
          return (
            <button key={item.id} className={`nav-item ${active ? "active" : ""}`} onClick={() => setPage(item.id)}>
              {active && <motion.span layoutId="active-nav" className="nav-active" />}
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="sidebar-status">
        <span>VISION ENGINE</span><b>OPENCV</b>
        <span>STATUS</span><b>ONLINE</b>
        <span>VERSION</span><b>V2.0</b>
      </div>
    </aside>
  );
}

function KpiCard({ label, value, unit, state }: { label: string; value: string; unit?: string; state?: "PASS" | "FAIL" }) {
  return (
    <motion.div className={`kpi ${state ? state.toLowerCase() : ""}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22 }}>
      <span>{label}</span>
      <strong className="mono">{value}</strong>
      {unit && <small>{unit}</small>}
    </motion.div>
  );
}

function InspectionViewport({
  result,
  selectedPreview,
  stage,
  setStage,
  inspecting,
  onDropFile,
}: {
  result: InspectionResult | null;
  selectedPreview?: string;
  stage: StageKey;
  setStage: (stage: StageKey) => void;
  inspecting: boolean;
  onDropFile: (file: File) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [drag, setDrag] = useState(false);
  const currentImage = result ? mediaUrl(result.images[stage]) : selectedPreview;
  const dimensions = result ? `${result.image_width} x ${result.image_height}` : "--";
  const isEdgeMapEmpty = stage === "edges" && result && result.metrics.num_defects === 0;

  // Reset zoom whenever a new result arrives
  useEffect(() => { setZoom(1); }, [result]);

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDrag(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onDropFile(file);
  };

  return (
    <section
      className={`viewport ${drag ? "dragging" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
    >
      <div className="panel-head">
        <div><b>INSPECTION VIEWPORT</b><span className="mono">{dimensions} PX</span></div>
        <div className="viewer-tools">
          <button onClick={() => setZoom(1)}>FIT</button>
          <button aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(3, z + 0.2))}><ZoomIn size={15} /></button>
          <button aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}><ZoomOut size={15} /></button>
          <button aria-label="Reset" onClick={() => setZoom(1)}><RotateCcw size={15} /></button>
          <span className="mono">{Math.round(zoom * 100)}%</span>
        </div>
      </div>
      <div className="stage-tabs">
        {stages.map((item) => <button className={stage === item.key ? "active" : ""} key={item.key} onClick={() => { setStage(item.key); setZoom(1); }}>{item.label}</button>)}
      </div>
      <div className="image-well" onWheel={(e) => { e.preventDefault(); setZoom((z) => Math.min(4, Math.max(0.5, z + (e.deltaY < 0 ? 0.1 : -0.1)))); }}>
        <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
        {inspecting && <><div className="scan-grid" /><div className="scan-line" /></>}
        <AnimatePresence mode="wait">
          {currentImage ? (
            <>
              <motion.img
                key={currentImage + stage}
                src={currentImage}
                style={{ transform: `scale(${zoom})` }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              />
              {isEdgeMapEmpty && (
                <div className="edge-empty-overlay">
                  <span>NO EDGES DETECTED</span>
                  <small>Good board — uniform surface, no structural anomalies</small>
                </div>
              )}
            </>
          ) : (
            <motion.div className="empty-view" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="idle-scan" />
              <ImageUp size={38} />
              <b>{drag ? "DROP IMAGE TO INSPECT" : "NO INSPECTION IMAGE"}</b>
              <span>SELECT A SAMPLE OR UPLOAD A PCB IMAGE</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}

function ParameterPanel({ params, setParams, save, setSave, run, disabled }: {
  params: InspectionParameters; setParams: (p: InspectionParameters) => void; save: boolean; setSave: (v: boolean) => void; run: () => void; disabled: boolean;
}) {
  const update = (key: keyof InspectionParameters, value: number) => setParams({ ...params, [key]: value });
  return (
    <aside className="controls">
      <div className="panel-head"><b>INSPECTION PARAMETERS</b><Settings size={17} /></div>
      <ControlGroup title="CANNY EDGE THRESHOLDS">
        <Slider label="LOWER THRESHOLD" value={params.canny_low} min={10} max={200} step={5} onChange={(v) => update("canny_low", v)} help="Lower hysteresis bound. Weaker edges are discarded." />
        <Slider label="UPPER THRESHOLD" value={params.canny_high} min={50} max={400} step={10} onChange={(v) => update("canny_high", v)} help="Upper hysteresis bound. Stronger edges are retained." />
      </ControlGroup>
      <ControlGroup title="CONTOUR FILTERING">
        <Slider label="MIN DEFECT AREA" value={params.min_defect_area} min={10} max={2000} step={10} suffix="PX2" onChange={(v) => update("min_defect_area", v)} help="Rejects contours below this area as micro-noise." />
      </ControlGroup>
      <ControlGroup title="QUALITY GATE">
        <Slider label="PASS / FAIL THRESHOLD" value={params.pass_fail_threshold} min={0.1} max={20} step={0.1} suffix="%" onChange={(v) => update("pass_fail_threshold", Number(v.toFixed(1)))} help="Maximum allowed defect coverage for PASS." />
      </ControlGroup>
      <ControlGroup title="PREPROCESSING">
        <div className="segmented">{[3, 5, 7, 9, 11].map((k) => <button key={k} className={params.blur_kernel === k ? "active" : ""} onClick={() => update("blur_kernel", k)}>{k}</button>)}</div>
      </ControlGroup>
      <label className="check"><input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} />SAVE INSPECTION OUTPUT</label>
      <button className="run" disabled={disabled} onClick={run}><Play size={17} />{disabled ? "INSPECTING" : "RUN INSPECTION"}</button>
    </aside>
  );
}

function ControlGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return <div className="control-group"><h3>{title}</h3>{children}</div>;
}

function Slider({ label, value, min, max, step, suffix = "", help, onChange }: {
  label: string; value: number; min: number; max: number; step: number; suffix?: string; help: string; onChange: (value: number) => void;
}) {
  return (
    <label className="slider" title={help}>
      <span>{label}<b className="mono">{value}{suffix}</b></span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function Processing({ active }: { active: boolean }) {
  const names = ["IMAGE ACQUIRED", "PREPROCESSING", "EDGE EXTRACTION", "CONTOUR ANALYSIS", "QUALITY DECISION"];
  return (
    <div className={`process ${active ? "active" : ""}`}>
      {names.map((name, i) => <div key={name} className="process-step"><span>{String(i + 1).padStart(2, "0")}</span><b>{name}</b></div>)}
    </div>
  );
}

function SampleSelector({ samples, selected, select }: { samples: SampleImage[]; selected: SampleImage | null; select: (s: SampleImage) => void }) {
  const [tab, setTab] = useState<"GOOD" | "DEFECTIVE">("GOOD");
  const filtered = samples.filter((s) => s.category === tab);
  return (
    <section className="samples">
      <div className="panel-head"><b>DEMO SAMPLES</b><div className="stage-tabs small"><button className={tab === "GOOD" ? "active" : ""} onClick={() => setTab("GOOD")}>GOOD BOARDS</button><button className={tab === "DEFECTIVE" ? "active" : ""} onClick={() => setTab("DEFECTIVE")}>DEFECTIVE BOARDS</button></div></div>
      <div className="sample-grid">
        {filtered.map((sample) => (
          <motion.button key={sample.category + sample.filename} className={`sample ${selected?.filename === sample.filename ? "selected" : ""}`} onClick={() => select(sample)} whileTap={{ y: 1 }}>
            <img src={mediaUrl(sample.preview_url)} />
            <span className="mono">{sample.filename}</span>
            <small>{sample.category}</small>
          </motion.button>
        ))}
        {!filtered.length && <div className="empty-panel">NO SAMPLE IMAGES</div>}
      </div>
    </section>
  );
}

function InspectionPage(props: {
  online: boolean; params: InspectionParameters; setParams: (p: InspectionParameters) => void; result: InspectionResult | null; setResult: (r: InspectionResult | null) => void; refreshData: () => Promise<void>; samples: SampleImage[];
}) {
  const { online, params, setParams, result, setResult, refreshData, samples } = props;
  const [file, setFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleImage | null>(null);
  const [stage, setStage] = useState<StageKey>("annotated");
  const [inspecting, setInspecting] = useState(false);
  const [save, setSave] = useState(true);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const selectedPreview = file ? URL.createObjectURL(file) : selectedSample ? mediaUrl(selectedSample.preview_url) : undefined;

  async function run() {
    if (!online) return setError("VISION ENGINE OFFLINE");
    if (!file && !selectedSample) return setError("INVALID IMAGE INPUT");
    setError("");
    setInspecting(true);
    try {
      const response = file ? await inspectImage(file, params, save) : await inspectSample(selectedSample!, params, save);
      setResult(response);
      setStage("annotated");
      await refreshData();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "INSPECTION REQUEST FAILED";
      setError(msg.length > 80 ? "INSPECTION REQUEST FAILED" : msg.toUpperCase());
    } finally {
      setInspecting(false);
    }
  }

  const metrics = result?.metrics;
  return (
    <PageFrame>
      {!online && <div className="banner offline">START FASTAPI BACKEND TO ENABLE INSPECTION</div>}
      {error && <div className="banner error">{error}</div>}
      <section className="kpi-strip">
        <KpiCard label="DEFECTS FOUND" value={metrics ? String(metrics.num_defects) : "--"} />
        <KpiCard label="TOTAL DEFECT AREA" value={metrics ? Math.round(metrics.total_defect_area).toLocaleString() : "--"} unit="PX2" />
        <KpiCard label="DEFECT COVERAGE" value={metrics ? metrics.defect_percentage.toFixed(4) : "--"} unit="%" />
        <KpiCard label="INSPECTION CONFIDENCE" value={metrics ? metrics.confidence.toFixed(1) : "--"} unit="%" />
        <KpiCard label="QUALITY VERDICT" value={result?.verdict ?? "--"} state={result?.verdict === "PASS" ? "PASS" : result?.verdict === "FAIL" ? "FAIL" : undefined} />
      </section>
      <main className="workspace">
        <InspectionViewport result={result} selectedPreview={selectedPreview} stage={stage} setStage={setStage} inspecting={inspecting} onDropFile={(f) => { setFile(f); setSelectedSample(null); setResult(null); }} />
        <ParameterPanel params={params} setParams={setParams} save={save} setSave={setSave} run={run} disabled={inspecting} />
      </main>
      <div className="action-row">
        <input ref={inputRef} type="file" className="hidden" accept="image/*" onChange={(e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) { setFile(f); setSelectedSample(null); setResult(null); } }} />
        <button onClick={() => inputRef.current?.click()}><Upload size={16} />UPLOAD IMAGE</button>
        <span className="mono">{file?.name ?? selectedSample?.filename ?? "READY FOR IMAGE SELECTION"}</span>
      </div>
      <Processing active={inspecting} />
      {result && <div className={`result-panel ${result.verdict.toLowerCase()}`}><ShieldCheck size={20} /><b>{result.verdict === "PASS" ? "QUALITY GATE PASSED" : "QUALITY GATE FAILED"}</b><span>ACTUAL COVERAGE <b className="mono">{result.metrics.defect_percentage}%</b></span><span>CONFIGURED THRESHOLD <b className="mono">{result.parameters.pass_fail_threshold}%</b></span><span>INSPECTION CONFIDENCE <b className="mono">{result.metrics.confidence}%</b></span></div>}
      <SampleSelector samples={samples} selected={selectedSample} select={(s) => { setSelectedSample(s); setFile(null); setResult(null); }} />
    </PageFrame>
  );
}

function PipelinePage({ result, params }: { result: InspectionResult | null; params: InspectionParameters }) {
  const [stage, setStage] = useState<StageKey>("annotated");
  const flow = [
    ["INPUT IMAGE", "Receives the uploaded or sample PCB image.", ""],
    ["GRAYSCALE CONVERSION", "Reduces color channels to luminance.", ""],
    ["GAUSSIAN BLUR", "Suppresses high-frequency noise.", `KERNEL ${params.blur_kernel}`],
    ["CANNY EDGE DETECTION", "Detects structural intensity transitions using dual-threshold hysteresis.", `LOW ${params.canny_low} HIGH ${params.canny_high}`],
    ["CONTOUR DETECTION", "Groups connected edge boundaries into candidate defect regions.", ""],
    ["AREA FILTERING", "Rejects contours below the configured minimum area to suppress micro-noise.", `MIN ${params.min_defect_area} PX2`],
    ["DEFECT LOCALIZATION", "Draws bounding boxes over retained defect candidates.", ""],
    ["QUALITY DECISION", "Compares total defect coverage against the configured quality threshold.", `THRESHOLD ${params.pass_fail_threshold}%`],
  ];
  return (
    <PageFrame>
      <div className="page-title"><h2>VISION PIPELINE</h2><p>DETERMINISTIC OPENCV INSPECTION FLOW</p></div>
      <div className="pipeline-grid">{flow.map((f, i) => <div className="pipe-card" key={f[0]}><span className="mono">{String(i + 1).padStart(2, "0")}</span><b>{f[0]}</b><p>{f[1]}</p>{f[2] && <small className="mono">{f[2]}</small>}</div>)}</div>
      <section className="pipeline-preview">
        <div className="stage-tabs">{stages.map((s) => <button key={s.key} className={stage === s.key ? "active" : ""} onClick={() => setStage(s.key)}>{s.label}</button>)}</div>
        {result ? <img src={mediaUrl(result.images[stage])} /> : <div className="empty-panel">RUN AN INSPECTION TO VIEW REAL PIPELINE ARTIFACTS</div>}
      </section>
    </PageFrame>
  );
}

function AnalyticsPage({ history }: { history: HistoryRecord[] }) {
  const pass = history.filter((h) => h.verdict === "PASS").length;
  const fail = history.filter((h) => h.verdict === "FAIL").length;
  const avgCoverage = history.length ? history.reduce((a, h) => a + h.defect_percentage, 0) / history.length : 0;
  const avgTime = history.length ? history.reduce((a, h) => a + h.inspection_duration_ms, 0) / history.length : 0;
  const trend = [...history].reverse().map((h, i) => ({ n: i + 1, coverage: h.defect_percentage, defects: h.num_defects, confidence: h.confidence, threshold: h.parameters.pass_fail_threshold }));
  return (
    <PageFrame>
      <section className="kpi-strip analytics-kpis">
        <KpiCard label="TOTAL INSPECTIONS" value={String(history.length)} />
        <KpiCard label="PASS" value={String(pass)} />
        <KpiCard label="FAIL" value={String(fail)} />
        <KpiCard label="PASS RATE" value={history.length ? ((pass / history.length) * 100).toFixed(1) : "--"} unit="%" />
        <KpiCard label="AVERAGE COVERAGE" value={history.length ? avgCoverage.toFixed(4) : "--"} unit="%" />
        <KpiCard label="AVERAGE TIME" value={history.length ? avgTime.toFixed(0) : "--"} unit="MS" />
      </section>
      {!history.length ? <div className="empty-panel tall">NO INSPECTION HISTORY</div> : (
        <div className="charts">
          <ChartPanel title="PASS VS FAIL"><ResponsiveContainer><PieChart><Pie data={[{ name: "PASS", value: pass }, { name: "FAIL", value: fail }]} innerRadius={55} outerRadius={82} dataKey="value"><Cell fill="#35d07f" /><Cell fill="#ff5c67" /></Pie><Tooltip /></PieChart></ResponsiveContainer></ChartPanel>
          <ChartPanel title="DEFECT COVERAGE TREND"><ResponsiveContainer><LineChart data={trend}><CartesianGrid stroke="#273544" /><XAxis dataKey="n" stroke="#778899" /><YAxis stroke="#778899" /><Tooltip /><Line dataKey="coverage" stroke="#2dd4ff" dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
          <ChartPanel title="DEFECT COUNT TREND"><ResponsiveContainer><AreaChart data={trend}><CartesianGrid stroke="#273544" /><XAxis dataKey="n" stroke="#778899" /><YAxis stroke="#778899" /><Tooltip /><Area dataKey="defects" stroke="#f2b84b" fill="#f2b84b33" /></AreaChart></ResponsiveContainer></ChartPanel>
          <ChartPanel title="QUALITY THRESHOLD COMPARISON"><ResponsiveContainer><LineChart data={trend}><CartesianGrid stroke="#273544" /><XAxis dataKey="n" stroke="#778899" /><YAxis stroke="#778899" /><Tooltip /><Line dataKey="coverage" stroke="#ff5c67" dot={false} /><Line dataKey="threshold" stroke="#35d07f" dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
        </div>
      )}
    </PageFrame>
  );
}

function ChartPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="chart-panel"><b>{title}</b><div>{children}</div></section>;
}

function HistoryPage({ history }: { history: HistoryRecord[] }) {
  const [selected, setSelected] = useState<HistoryRecord | null>(null);
  return (
    <PageFrame>
      <table className="history-table"><thead><tr>{["INSPECTION ID", "TIMESTAMP", "IMAGE", "DEFECTS", "DEFECT AREA", "COVERAGE", "CONFIDENCE", "RESULT", "DURATION"].map((h) => <th key={h}>{h}</th>)}</tr></thead><tbody>{history.map((h) => <tr key={h.inspection_id} onClick={() => setSelected(h)}><td className="mono">{h.inspection_id}</td><td>{h.timestamp}</td><td>{h.filename}</td><td>{h.num_defects}</td><td>{Math.round(h.total_defect_area).toLocaleString()}</td><td>{h.defect_percentage}%</td><td>{h.confidence}%</td><td><span className={`status ${h.verdict.toLowerCase()}`}>{h.verdict}</span></td><td>{h.inspection_duration_ms}MS</td></tr>)}</tbody></table>
      {!history.length && <div className="empty-panel tall">NO HISTORY RECORDS</div>}
      <AnimatePresence>{selected && <motion.aside className="drawer" initial={{ x: 420 }} animate={{ x: 0 }} exit={{ x: 420 }} transition={{ duration: 0.25 }}><button className="drawer-close" onClick={() => setSelected(null)}>CLOSE</button><h2 className="mono">{selected.inspection_id}</h2><p>{selected.filename}</p><img src={mediaUrl(selected.images.annotated)} /><div className="metric-list"><span>DEFECTS <b>{selected.num_defects}</b></span><span>COVERAGE <b>{selected.defect_percentage}%</b></span><span>CONFIDENCE <b>{selected.confidence}%</b></span><span>RESULT <b className={selected.verdict.toLowerCase()}>{selected.verdict}</b></span></div><pre>{JSON.stringify(selected.parameters, null, 2)}</pre></motion.aside>}</AnimatePresence>
    </PageFrame>
  );
}

function SystemPage({ params, online }: { params: InspectionParameters; online: boolean }) {
  return (
    <PageFrame>
      <div className="system-grid">
        {[
          ["VISION ENGINE", "OpenCV"],
          ["PIPELINE TYPE", "Classical Computer Vision"],
          ["INSPECTION METHOD", "Canny Edge Detection + Contour Analysis"],
          ["QUALITY DECISION", "Defect Coverage Threshold"],
          ["BACKEND", "FastAPI"],
          ["FRONTEND", "React + TypeScript"],
          ["SYSTEM STATUS", online ? "ONLINE" : "OFFLINE"],
          ["API BASE", API_BASE],
        ].map(([k, v]) => <div className="sys-card" key={k}><span>{k}</span><b>{v}</b></div>)}
      </div>
      <section className="text-panel"><h2>ABOUT THE INSPECTION ENGINE</h2><p>VisionInspect AI uses deterministic classical computer vision. The pipeline converts the image to grayscale, applies Gaussian smoothing, extracts structural boundaries using Canny edge detection, identifies candidate regions using contours, filters micro-noise using area thresholds, and calculates total defect coverage for the quality decision.</p></section>
      <section className="text-panel"><h2>CURRENT PARAMETERS</h2><pre>{JSON.stringify(params, null, 2)}</pre></section>
      <section className="text-panel"><h2>CURRENT LIMITATIONS</h2><p>The current detector is sensitive to lighting, camera angle, product geometry, and threshold configuration.</p></section>
      <section className="text-panel"><h2>NEXT INSPECTION ENGINE</h2><p>Roadmap: ROI-based PCB component presence inspection, template matching, ORB feature matching, position validation, and component count validation.</p></section>
    </PageFrame>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("inspection");
  const [online, setOnline] = useState(false);
  const [params, setParams] = useState(defaultParams);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  const refreshData = async () => {
    const [sampleData, historyData] = await Promise.all([getSamples().catch(() => []), getHistory().catch(() => [])]);
    setSamples(sampleData);
    setHistory(historyData);
  };

  useEffect(() => {
    getHealth().then(() => setOnline(true)).catch(() => setOnline(false));
    refreshData();
  }, []);

  const pageContent = useMemo(() => {
    if (page === "inspection") return <InspectionPage online={online} params={params} setParams={setParams} result={result} setResult={setResult} refreshData={refreshData} samples={samples} />;
    if (page === "pipeline") return <PipelinePage result={result} params={params} />;
    if (page === "analytics") return <AnalyticsPage history={history} />;
    if (page === "history") return <HistoryPage history={history} />;
    return <SystemPage params={params} online={online} />;
  }, [page, online, params, result, samples, history]);

  return (
    <div className="app">
      <Sidebar page={page} setPage={setPage} />
      <div className="main">
        <SystemBar online={online} />
        <AnimatePresence mode="wait">{pageContent}</AnimatePresence>
      </div>
    </div>
  );
}

