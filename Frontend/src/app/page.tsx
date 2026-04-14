"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import FileUploader from "@/components/FileUploader";
import axios from "axios";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Step = "idle" | "training" | "clustering" | "voronoi" | "done" | "error";
type Tab = "gmm" | "voronoi" | "heatmap" | "particle";

interface LatentRange { min: number; max: number; }
interface LatentConfig { latent_dim: number; latent_ranges: LatentRange[]; }
interface ClassInfo {
    predicted_class: string;
    confidence: number;
    class_distribution: Record<string, number>;
}

export default function App() {
    const [uploaded, setUploaded] = useState(false);
    const [step, setStep] = useState<Step>("idle");
    const [gmmPlot, setGmmPlot] = useState<any>(null);
    const [voronoiPlot, setVoronoiPlot] = useState<any>(null);
    const [heatmapImg, setHeatmapImg] = useState<string | null>(null);
    const [particlePlot, setParticlePlot] = useState<any>(null);
    const [targetCol, setTargetCol] = useState("Cover_Type");
    const [logs, setLogs] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState<Tab>("gmm");
    const [latentConfig, setLatentConfig] = useState<LatentConfig | null>(null);
    const [zValues, setZValues] = useState<number[]>([0, 0, 0]);
    const [latentLoading, setLatentLoading] = useState(false);
    const [classInfo, setClassInfo] = useState<ClassInfo | null>(null);
    const [selectedPoint, setSelectedPoint] = useState<number[] | null>(null);

    function addLog(msg: string) {
        setLogs(prev => [...prev, msg]);
    }

    function setZ(i: number, val: number) {
        setZValues(prev => { const next = [...prev]; next[i] = val; return next; });
    }

    function onClickGmm(eventData: any) {
        if (!eventData?.points?.length) return;
        const point = eventData.points[0];

        // Only respond to clicks on data points (trace 0), not centroids
        if (point.curveNumber !== 0) return;

        if (point.customdata) {
            const coords = Array.isArray(point.customdata)
                ? point.customdata
                : [point.x, point.y, point.z];

            const z1 = coords[0] ?? 0;
            const z2 = coords[1] ?? 0;
            const z3 = coords[2] ?? 0;

            setZValues(prev => {
                const next = [...prev];
                next[0] = z1;
                if (next.length > 1) next[1] = z2;
                if (next.length > 2) next[2] = z3;
                return next;
            });

            setSelectedPoint([z1, z2, z3]);
            addLog(`Point selected — Z1: ${z1.toFixed(2)}, Z2: ${z2.toFixed(2)}, Z3: ${z3.toFixed(2)}`);
            addLog(`Switch to Heatmap or Particle tab and click Generate`);
        }
    }

    async function handleRunPipeline() {
        if (!uploaded) return;
        setGmmPlot(null);
        setVoronoiPlot(null);
        setClassInfo(null);
        setHeatmapImg(null);
        setParticlePlot(null);
        setSelectedPoint(null);
        setLogs([]);

        try {
            addLog("Starting VAE training...");
            setStep("training");
            await axios.post("http://localhost:8000/vae_training");
            addLog("VAE done.");
        } catch (e: any) {
            addLog(`VAE FAILED: ${e?.response?.data?.detail ?? e?.message}`);
            setStep("error");
            return;
        }

        try {
            addLog("Starting GMM clustering...");
            setStep("clustering");
            const gmmRes = await axios.post("http://localhost:8000/GMM_bic");
            setGmmPlot(gmmRes.data);
            addLog("GMM done.");
        } catch (e: any) {
            addLog(`GMM FAILED: ${e?.response?.data?.detail ?? e?.message}`);
            setStep("error");
            return;
        }

        try {
            addLog("Running UMAP + Voronoi...");
            setStep("voronoi");
            const vorRes = await axios.get("http://localhost:8000/voronoi");
            setVoronoiPlot(vorRes.data);
            addLog("Voronoi done.");
        } catch (e: any) {
            addLog(`Voronoi FAILED: ${e?.response?.data?.detail ?? e?.message}`);
            setStep("error");
            return;
        }

        try {
            const cfgRes = await axios.get("http://localhost:8000/config");
            const cfg: LatentConfig = cfgRes.data;
            setLatentConfig(cfg);
            setZValues(cfg.latent_ranges.map(r => (r.min + r.max) / 2));
            addLog(`Latent dim: ${cfg.latent_dim}`);
        } catch (e: any) {
            addLog(`Config fetch failed: ${e?.message}`);
        }

        setStep("done");
    }

    async function fetchLatentViz() {
        if (!latentConfig) return;
        setLatentLoading(true);
        try {
            const z1 = zValues[0] ?? 0;
            const z2 = zValues[1] ?? 0;
            const z3 = zValues[2] ?? 0;

            if (activeTab === "heatmap") {
                const res = await axios.get(`http://localhost:8000/heatmap?z1=${z1}&z2=${z2}&z3=${z3}`);
                setHeatmapImg(res.data.image);
                setClassInfo(res.data.class_info);
            } else if (activeTab === "particle") {
                const res = await axios.get(`http://localhost:8000/particle?z1=${z1}&z2=${z2}&z3=${z3}`);
                setParticlePlot(res.data);
                setClassInfo(res.data.class_info);
            }
        } catch (e: any) {
            addLog(`Latent viz FAILED: ${e?.response?.data?.error ?? e?.message}`);
        }
        setLatentLoading(false);
    }

    const stepLabel: Record<Step, string> = {
        idle: "Run Pipeline", training: "Training VAE...",
        clustering: "Clustering...", voronoi: "Running Voronoi...",
        done: "Run Again", error: "Retry",
    };

    const isRunning = ["training", "clustering", "voronoi"].includes(step);
    const pipelineDone = step === "done";
    const showSliders = pipelineDone && (activeTab === "heatmap" || activeTab === "particle") && latentConfig;

    const CLASS_COLOURS = [
        "#4ade80", "#60a5fa", "#f87171", "#fbbf24",
        "#a78bfa", "#34d399", "#fb923c", "#e879f9"
    ];

    const tabs: { key: Tab; label: string }[] = [
        { key: "gmm", label: "GMM" },
        { key: "voronoi", label: "Voronoi" },
        { key: "heatmap", label: "Heatmap" },
        { key: "particle", label: "Particle" },
    ];

    return (
        <main className="min-h-screen p-5 bg-[#F0EBE1]">
            <div className="mb-5 flex items-center">
                <h4 className="font-bold text-sm text-[#0D0D0D]">Welcome Back</h4>
                <h1 className="font-bold text-xl text-[#0D0D0D] absolute left-1/2 -translate-x-1/2">
                    Data Visualisation
                </h1>
            </div>

            <div className="grid grid-cols-[1fr_3fr] gap-4">
                {/* Left panel */}
                <div className="bg-[#E8E0D0] rounded-xl border-2 border-[#0D0D0D] flex flex-col gap-4 p-4">
                    <span className="text-sm text-[#7A7060] text-center">Upload File Here</span>

                    <div className="w-full flex flex-col gap-1">
                        <label className="text-xs text-[#7A7060]">Target Column</label>
                        <input
                            type="text" value={targetCol}
                            onChange={(e) => setTargetCol(e.target.value)}
                            className="border border-[#0D0D0D] rounded px-2 py-1 text-sm w-full bg-transparent text-black"
                            placeholder="e.g. Cover_Type"
                        />
                    </div>

                    <FileUploader
                        targetCol={targetCol}
                        onUploaded={() => { setUploaded(true); addLog("CSV uploaded successfully"); }}
                    />

                    {uploaded && (
                        <button
                            onClick={handleRunPipeline} disabled={isRunning}
                            className="w-full bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg disabled:opacity-50"
                        >
                            {stepLabel[step]}
                        </button>
                    )}

                    <div className="text-xs flex flex-col gap-1 w-full">
                        <p className={uploaded ? "text-green-700" : "text-[#7A7060]"}>{uploaded ? "✓" : "○"} 1. Upload CSV</p>
                        <p className={step !== "idle" && step !== "error" ? "text-green-700" : "text-[#7A7060]"}>{step !== "idle" && step !== "error" ? "✓" : "○"} 2. Train VAE</p>
                        <p className={["clustering", "voronoi", "done"].includes(step) ? "text-green-700" : "text-[#7A7060]"}>{["clustering", "voronoi", "done"].includes(step) ? "✓" : "○"} 3. GMM</p>
                        <p className={["voronoi", "done"].includes(step) ? "text-green-700" : "text-[#7A7060]"}>{["voronoi", "done"].includes(step) ? "✓" : "○"} 4. Voronoi</p>
                    </div>

                    {/* Selected point indicator */}
                    {selectedPoint && (
                        <div className="bg-[#0D0D0D] text-[#F0EBE1] rounded-lg px-3 py-2 text-xs">
                            <p className="font-bold mb-1">📍 Selected point</p>
                            <p>Z1: {selectedPoint[0].toFixed(3)}</p>
                            <p>Z2: {selectedPoint[1].toFixed(3)}</p>
                            <p>Z3: {selectedPoint[2].toFixed(3)}</p>
                            <p className="text-[#7A7060] mt-1">Switch to Heatmap or Particle and click Generate</p>
                        </div>
                    )}

                    {/* Dynamic latent sliders */}
                    {showSliders && (
                        <div className="flex flex-col gap-2 w-full">
                            <p className="text-xs font-bold text-[#0D0D0D]">
                                Latent Space
                                <span className="font-normal text-[#7A7060] ml-1">
                                    ({latentConfig.latent_dim} dims, controlling first 3)
                                </span>
                            </p>
                            {latentConfig.latent_ranges.slice(0, 3).map((range, i) => (
                                <div key={i} className="flex flex-col gap-1">
                                    <label className="text-xs text-[#7A7060]">
                                        Z{i + 1}: {(zValues[i] ?? 0).toFixed(2)}
                                        <span className="ml-1 text-[#9A8A70]">
                                            ({range.min.toFixed(1)} → {range.max.toFixed(1)})
                                        </span>
                                    </label>
                                    <input
                                        type="range"
                                        min={range.min} max={range.max}
                                        step={(range.max - range.min) / 100}
                                        value={zValues[i] ?? 0}
                                        onChange={(e) => setZ(i, parseFloat(e.target.value))}
                                        className="w-full"
                                    />
                                </div>
                            ))}
                            <button
                                onClick={fetchLatentViz} disabled={latentLoading}
                                className="w-full bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg disabled:opacity-50 text-sm"
                            >
                                {latentLoading ? "Generating..." : "Generate"}
                            </button>
                        </div>
                    )}

                    {logs.length > 0 && (
                        <div className="bg-[#0D0D0D] text-green-400 rounded p-2 text-xs font-mono flex flex-col gap-1 max-h-40 overflow-y-auto">
                            {logs.map((l, i) => <span key={i}>{l}</span>)}
                        </div>
                    )}
                </div>

                {/* Right panel */}
                <div className="bg-[#C8B4A0] rounded-xl p-4 flex flex-col gap-3 min-h-[600px]">
                    {pipelineDone && (
                        <div className="flex gap-2">
                            {tabs.map(t => (
                                <button key={t.key} onClick={() => setActiveTab(t.key)}
                                    className={`py-1 px-4 rounded-lg text-sm ${activeTab === t.key ? "bg-[#0D0D0D] text-[#F0EBE1]" : "bg-[#E8E0D0]"}`}>
                                    {t.label}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Class prediction banner */}
                    {pipelineDone && (activeTab === "heatmap" || activeTab === "particle") && classInfo && (
                        <div className="bg-[#0D0D0D] text-[#F0EBE1] rounded-xl px-5 py-3 flex flex-col gap-2">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-[#7A7060]">Most likely class</p>
                                    <p className="text-2xl font-bold">{classInfo.predicted_class}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-[#7A7060]">Confidence</p>
                                    <p className="text-2xl font-bold">{(classInfo.confidence * 100).toFixed(0)}%</p>
                                </div>
                            </div>
                            <div>
                                <p className="text-xs text-[#7A7060] mb-1">Nearby class mix</p>
                                <div className="flex h-4 rounded overflow-hidden w-full">
                                    {Object.entries(classInfo.class_distribution).map(([cls, prob], i) => (
                                        <div key={cls}
                                            style={{ width: `${prob * 100}%`, backgroundColor: CLASS_COLOURS[i % CLASS_COLOURS.length] }}
                                            title={`${cls}: ${(prob * 100).toFixed(0)}%`}
                                        />
                                    ))}
                                </div>
                                <div className="flex flex-wrap gap-x-3 mt-1">
                                    {Object.entries(classInfo.class_distribution).map(([cls, prob], i) => (
                                        <span key={cls} className="text-xs flex items-center gap-1">
                                            <span style={{ backgroundColor: CLASS_COLOURS[i % CLASS_COLOURS.length] }}
                                                className="inline-block w-2 h-2 rounded-full" />
                                            {cls} {(prob * 100).toFixed(0)}%
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="flex-1 flex items-center justify-center">
                        {isRunning && <p className="text-[#0D0D0D]">{stepLabel[step]}</p>}
                        {!isRunning && !pipelineDone && (
                            <p className="text-[#7A7060] text-sm">Run the pipeline to see plots</p>
                        )}

                        {/* GMM */}
                        {pipelineDone && activeTab === "gmm" && gmmPlot && (
                            <div style={{ width: "100%", backgroundColor: "white", borderRadius: "8px" }}>
                                <p className="text-xs text-[#7A7060] text-center py-2">
                                    💡 Click any point to load it into the Heatmap &amp; Particle sliders
                                </p>
                                <div style={{ height: "520px" }}>
                                    <Plot
                                        data={gmmPlot.data}
                                        layout={{ ...gmmPlot.layout, autosize: true, width: undefined, height: undefined, paper_bgcolor: "white" }}
                                        style={{ width: "100%", height: "100%" }}
                                        useResizeHandler
                                        config={{ responsive: true, scrollZoom: true }}
                                        onClick={onClickGmm}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Voronoi */}
                        {pipelineDone && activeTab === "voronoi" && voronoiPlot && (
                            <div style={{ width: "100%", height: "560px", backgroundColor: "white", borderRadius: "8px" }}>
                                <Plot
                                    data={voronoiPlot.data}
                                    layout={{ ...voronoiPlot.layout, autosize: true, width: undefined, height: undefined, paper_bgcolor: "white", plot_bgcolor: "white" }}
                                    style={{ width: "100%", height: "100%" }}
                                    useResizeHandler
                                    config={{ responsive: true }}
                                />
                            </div>
                        )}

                        {/* Heatmap */}
                        {pipelineDone && activeTab === "heatmap" && (
                            <div className="w-full flex items-center justify-center">
                                {heatmapImg
                                    ? <img src={heatmapImg} alt="Heatmap" className="w-full rounded-lg" />
                                    : <p className="text-[#7A7060] text-sm">Click a point on the GMM plot or set Z values, then click Generate</p>
                                }
                            </div>
                        )}

                        {/* Particle */}
                        {pipelineDone && activeTab === "particle" && (
                            <div style={{ width: "100%", height: "560px", backgroundColor: "white", borderRadius: "8px" }}>
                                {particlePlot
                                    ? <Plot
                                        data={particlePlot.data}
                                        layout={{ ...particlePlot.layout, autosize: true, width: undefined, height: undefined, paper_bgcolor: "white" }}
                                        style={{ width: "100%", height: "100%" }}
                                        useResizeHandler
                                        config={{ responsive: true }}
                                    />
                                    : <div className="h-full flex items-center justify-center">
                                        <p className="text-[#7A7060] text-sm">Click a point on the GMM plot or set Z values, then click Generate</p>
                                    </div>
                                }
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}