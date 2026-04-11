"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import FileUploader from "@/components/FileUploader";
import axios from "axios";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Step = "idle" | "training" | "clustering" | "done" | "error";

export default function App() {
  const [uploaded, setUploaded] = useState(false);
  const [step, setStep] = useState<Step>("idle");
  const [gmmPlot, setGmmPlot] = useState<any>(null);
  const [targetCol, setTargetCol] = useState("Cover_Type");
  const [logs, setLogs] = useState<string[]>([]);

  function addLog(msg: string) {
    console.log(msg);
    setLogs(prev => [...prev, msg]);
  }

  async function handleRunPipeline() {
    if (!uploaded) return;
    setGmmPlot(null);
    setLogs([]);

    try {
      addLog("Starting VAE training...");
      setStep("training");
      const vaeRes = await axios.post("http://localhost:8000/vae_training");
      addLog(`VAE done. Status: ${vaeRes.status}`);
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? e?.message ?? String(e);
      addLog(`VAE FAILED: ${msg}`);
      setStep("error");
      return;
    }

    try {
      addLog("Starting GMM clustering...");
      setStep("clustering");
      const gmmRes = await axios.post("http://localhost:8000/GMM_bic");
      addLog(`GMM done. Status: ${gmmRes.status}`);
      addLog(`Response keys: ${Object.keys(gmmRes.data).join(", ")}`);
      addLog(`Data traces: ${gmmRes.data?.data?.length ?? "none"}`);
      setGmmPlot(gmmRes.data);
      setStep("done");
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? e?.message ?? String(e);
      addLog(`GMM FAILED: ${msg}`);
      setStep("error");
      return;
    }
  }

  const stepLabel: Record<Step, string> = {
    idle: "Run Pipeline",
    training: "Training VAE...",
    clustering: "Clustering...",
    done: "Run Again",
    error: "Retry",
  };

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
              type="text"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="border border-[#0D0D0D] rounded px-2 py-1 text-sm w-full bg-transparent"
              placeholder="e.g. Cover_Type"
            />
          </div>

          <FileUploader
            targetCol={targetCol}
            onUploaded={() => {
              setUploaded(true);
              addLog("CSV uploaded successfully");
            }}
          />

          {uploaded && (
            <button
              onClick={handleRunPipeline}
              disabled={step === "training" || step === "clustering"}
              className="w-full bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg disabled:opacity-50"
            >
              {stepLabel[step]}
            </button>
          )}

          <div className="text-xs flex flex-col gap-1 w-full">
            <p className={uploaded ? "text-green-700" : "text-[#7A7060]"}>
              {uploaded ? "✓" : "○"} 1. Upload CSV
            </p>
            <p className={step !== "idle" && step !== "error" ? "text-green-700" : "text-[#7A7060]"}>
              {step !== "idle" && step !== "error" ? "✓" : "○"} 2. Train VAE
            </p>
            <p className={step === "clustering" || step === "done" ? "text-green-700" : "text-[#7A7060]"}>
              {step === "clustering" || step === "done" ? "✓" : "○"} 3. GMM Clustering
            </p>
          </div>

          {logs.length > 0 && (
            <div className="bg-[#0D0D0D] text-green-400 rounded p-2 text-xs font-mono flex flex-col gap-1 max-h-40 overflow-y-auto">
              {logs.map((l, i) => <span key={i}>{l}</span>)}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="bg-[#C8B4A0] rounded-xl p-4 flex items-center justify-center min-h-[600px]">
          {(step === "idle" || step === "error") && (
            <p className="text-[#7A7060] text-sm">Run the pipeline to see the GMM plot</p>
          )}

          {step === "training" && (
            <p className="text-[#0D0D0D]">Training VAE... this may take a few minutes</p>
          )}

          {step === "clustering" && (
            <p className="text-[#0D0D0D]">Running GMM clustering...</p>
          )}

          {step === "done" && gmmPlot && (
            <div style={{ width: "100%", height: "560px", backgroundColor: "white", borderRadius: "8px" }}>
              <Plot
                data={gmmPlot.data}
                layout={{
                  ...gmmPlot.layout,
                  autosize: true,
                  width: undefined,
                  height: undefined,
                  paper_bgcolor: "white",
                  plot_bgcolor: "white",
                }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
                config={{
                  responsive: true,
                  scrollZoom: true,
                }}
              />
            </div>
          )}

          {step === "done" && !gmmPlot && (
            <p className="text-red-600 text-sm">
              Step completed but plot data was empty — check logs
            </p>
          )}
        </div>
      </div>
    </main>
  );
}