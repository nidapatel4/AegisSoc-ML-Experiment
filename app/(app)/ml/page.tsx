"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  Loader2,
  AlertTriangle,
  Sparkles,
  ShieldAlert,
  ShieldCheck,
  ArrowRight,
} from "lucide-react";
import { PageHeader, Panel, SeverityBadge, riskTone } from "@/components/app/primitives";
import { ScoreRing } from "@/components/app/score-ring";
import { usePageTitle } from "@/lib/use-page-title";
import type { MlAnalyzeResult, MlModelInfo } from "@/lib/ml-service";

// Map the ML service's risk buckets onto the console's Severity tones.
const levelToSeverity = {
  Critical: "Critical",
  High: "High",
  Medium: "Medium",
  Low: "Low",
} as const;

// NSL-KDD-style event presets. The "suspicious" preset is a brute-force /
// SYN-flood pattern (flag S0, failed logins, 100% error rates); the "normal"
// preset is a plain successful HTTP connection.
const SUSPICIOUS_EVENT = {
  duration: 0,
  protocol_type: "tcp",
  service: "private",
  flag: "S0",
  src_bytes: 0,
  dst_bytes: 0,
  count: 123,
  srv_count: 6,
  serror_rate: 1.0,
  srv_serror_rate: 1.0,
  same_srv_rate: 0.05,
  dst_host_count: 255,
  dst_host_srv_count: 26,
  dst_host_same_srv_rate: 0.05,
  num_failed_logins: 5,
  logged_in: 0,
  source_ip: "203.0.113.66",
  destination: "WEB-SERVER-01",
  event_id: "evt-suspicious",
};

const NORMAL_EVENT = {
  duration: 0,
  protocol_type: "tcp",
  service: "http",
  flag: "SF",
  src_bytes: 232,
  dst_bytes: 8153,
  count: 1,
  srv_count: 1,
  serror_rate: 0,
  srv_serror_rate: 0,
  same_srv_rate: 1.0,
  dst_host_count: 1,
  dst_host_srv_count: 1,
  dst_host_same_srv_rate: 1.0,
  num_failed_logins: 0,
  logged_in: 1,
  source_ip: "198.51.100.24",
  destination: "WEB-SERVER-01",
  event_id: "evt-normal",
};

const PORTSCAN_EVENT = {
  duration: 0,
  protocol_type: "tcp",
  service: "eco_i",
  flag: "S0",
  src_bytes: 0,
  dst_bytes: 0,
  count: 240,
  srv_count: 240,
  serror_rate: 0.98,
  srv_serror_rate: 0.98,
  same_srv_rate: 1.0,
  diff_srv_rate: 0,
  dst_host_count: 250,
  dst_host_srv_count: 250,
  dst_host_same_srv_rate: 1.0,
  dst_host_serror_rate: 0.97,
  num_failed_logins: 0,
  logged_in: 0,
  source_ip: "203.0.113.9",
  destination: "EDGE-FW-02",
  event_id: "evt-portscan",
};

const BATCH_EVENTS = [SUSPICIOUS_EVENT, PORTSCAN_EVENT, NORMAL_EVENT];

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function scoreTone(score: number): "signal" | "caution" | "clearance" {
  if (score >= 70) return "signal";
  if (score >= 40) return "caution";
  return "clearance";
}

export default function MlPage() {
  usePageTitle("ML Detection — AegisSOC");

  const [info, setInfo] = useState<MlModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);

  const [result, setResult] = useState<MlAnalyzeResult | null>(null);
  const [batch, setBatch] = useState<MlAnalyzeResult[] | null>(null);
  const [busy, setBusy] = useState<"single" | "batch" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/ml/info")
      .then(async (res) => {
        const body = (await res.json()) as MlModelInfo | { error: string };
        if ("error" in body) throw new Error(body.error);
        setInfo(body);
      })
      .catch((err: Error) => setInfoError(err.message));
  }, []);

  const metrics = info?.test_metrics;
  const metricRows = useMemo(
    () =>
      metrics
        ? [
            { label: "ROC-AUC", value: metrics.roc_auc },
            { label: "PR-AUC", value: metrics.pr_auc },
            { label: "Precision", value: metrics.precision },
            { label: "Recall", value: metrics.recall },
            { label: "F1", value: metrics.f1 },
            {
              label: "Novel-attack recall",
              value: metrics.recall_on_novel_unseen_attack_types,
            },
          ]
        : [],
    [metrics],
  );

  async function analyze(
    event: Record<string, unknown> | null,
    mode: "single" | "batch",
  ) {
    setError(null);
    setBusy(mode);
    try {
      const res = await fetch(mode === "single" ? "/api/ml/analyze" : "/api/ml/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "single" ? { event } : { events: BATCH_EVENTS }),
      });
      const body = (await res.json()) as
        | (MlAnalyzeResult | MlAnalyzeResult[])
        | { error: string };
      if ("error" in body) throw new Error(body.error);
      if (mode === "single") {
        setResult(body as MlAnalyzeResult);
        setBatch(null);
      } else {
        setBatch(body as MlAnalyzeResult[]);
        setResult(null);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "The analysis failed.");
    } finally {
      setBusy(null);
    }
  }

  const resultSeverity = result ? levelToSeverity[result.risk_level] : null;

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Machine Learning" title="ML Anomaly Detection">
        <span
          className={`inline-flex items-center gap-2 border px-3 py-1.5 font-mono text-[11px] ${
            info
              ? "border-line text-ink-soft dark:border-line-dark dark:text-paper/60"
              : "border-line text-ink-faint dark:border-line-dark dark:text-paper/40"
          }`}
        >
          <BrainCircuit size={13} className={info ? "text-clearance" : ""} />
          {info ? "Isolation Forest · NSL-KDD" : infoError ? "Service offline" : "Connecting…"}
        </span>
      </PageHeader>

      {infoError && (
        <div className="flex items-start gap-3 border border-signal/50 bg-signal-soft p-4 dark:bg-signal/10">
          <AlertTriangle size={16} className="mt-0.5 flex-none text-signal" />
          <div className="space-y-1">
            <p className="font-mono text-[12px] text-signal">ML service unreachable</p>
            <p className="font-mono text-[11px] text-ink-soft dark:text-paper/60">{infoError}</p>
          </div>
        </div>
      )}

      {/* model status + metrics */}
      <div className="grid gap-5 lg:grid-cols-3">
        <Panel label="Model card" className="lg:col-span-2">
          {info ? (
            <div className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                {[
                  ["Algorithm", info.model_type],
                  ["Training mode", info.training_mode],
                  ["Dataset", info.dataset],
                  [
                    "Operating threshold",
                    `${info.operating_threshold.toFixed(4)} @ ${pct(
                      info.target_false_positive_rate,
                    )} FPR`,
                  ],
                ].map(([label, value]) => (
                  <div key={label} className="border border-line p-3 dark:border-line-dark">
                    <div className="font-mono text-[9.5px] uppercase tracking-wideish text-ink-faint dark:text-paper/45">
                      {label}
                    </div>
                    <div className="mt-1.5 font-mono text-[12px] text-ink dark:text-paper">
                      {value}
                    </div>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-px border border-line bg-line dark:border-line-dark dark:bg-line-dark sm:grid-cols-6">
                {metricRows.map((m) => (
                  <div key={m.label} className="bg-paper p-3 dark:bg-void">
                    <div className="font-mono text-[9px] uppercase tracking-wideish text-ink-faint dark:text-paper/45">
                      {m.label}
                    </div>
                    <div className="mt-1 font-mono text-[15px] font-600 tabular-nums text-ink dark:text-paper">
                      {pct(m.value)}
                    </div>
                  </div>
                ))}
              </div>
              <p className="font-mono text-[10.5px] leading-relaxed text-ink-faint dark:text-paper/40">
                Evaluated on the untouched NSL-KDD test set (22,544 records), which contains 17
                attack types never seen during training — recall on those novel types is{" "}
                {pct(metrics!.recall_on_novel_unseen_attack_types)}, confirming the model detects
                deviation from normal behaviour rather than memorizing signatures. Known honest
                limitation: R2L/U2R recall is weak — pair with rule-based detection for credential
                attacks.
              </p>
            </div>
          ) : (
            <div className="flex h-32 items-center justify-center font-mono text-[11px] text-ink-faint dark:text-paper/40">
              {infoError ? "No model data." : "Loading model card…"}
            </div>
          )}
        </Panel>

        <Panel label="How it works">
          <div className="space-y-3 font-mono text-[11px] leading-relaxed text-ink-soft dark:text-paper/60">
            <p>
              The model is{" "}
              <span className="text-ink dark:text-paper">never shown a single attack</span> during
              training. It learns what normal traffic looks like, then flags deviation —
              mirroring how a real SOC operates.
            </p>
            <div className="space-y-2 border-l-2 border-line pl-3 dark:border-line-dark">
              {[
                "Real NSL-KDD baseline → preprocessing (fit on normal only)",
                "Isolation Forest · 300 trees",
                "Threshold calibrated @ 5% FPR on held-out normal",
                "Risk score 0–100 → Low / Medium / High / Critical",
                "Reason codes: why was this flagged",
              ].map((step, i) => (
                <div key={step} className="flex items-start gap-2">
                  <span className="font-600 text-signal">{i + 1}.</span>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* live analyzer */}
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel label="Event analyzer — score a security event">
          <div className="space-y-4">
            <p className="font-mono text-[11px] leading-relaxed text-ink-soft dark:text-paper/60">
              Score a normalized security event through the live model. Pick a preset or edit the
              JSON sent to <span className="text-signal">POST /api/ml/analyze</span>.
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => analyze(SUSPICIOUS_EVENT, "single")}
                disabled={busy !== null}
                className="inline-flex items-center gap-2 bg-signal px-3 py-2 font-mono text-[11px] uppercase tracking-wideish text-paper transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {busy === "single" ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                Analyze suspicious event
              </button>
              <button
                onClick={() => analyze(NORMAL_EVENT, "single")}
                disabled={busy !== null}
                className="inline-flex items-center gap-2 border border-line px-3 py-2 font-mono text-[11px] uppercase tracking-wideish text-ink-soft transition-colors hover:border-signal hover:text-signal dark:border-line-dark dark:text-paper/60 disabled:opacity-50"
              >
                <ShieldCheck size={13} />
                Analyze normal event
              </button>
            </div>
            <pre className="max-h-44 overflow-auto border border-line bg-paper-dim p-3 font-mono text-[10px] leading-relaxed text-ink-soft dark:border-line-dark dark:bg-void-surface dark:text-paper/60">
              {JSON.stringify(SUSPICIOUS_EVENT, null, 2)}
            </pre>
            {error && (
              <div className="flex items-start gap-2 border border-signal/50 bg-signal-soft p-3 dark:bg-signal/10">
                <AlertTriangle size={14} className="mt-0.5 flex-none text-signal" />
                <span className="font-mono text-[10.5px] text-signal">{error}</span>
              </div>
            )}
          </div>
        </Panel>

        <Panel
          label="Model verdict"
          right={result && <SeverityBadge level={resultSeverity ?? "Low"} />}
        >
          {result ? (
            <div className="space-y-4">
              <div className="flex items-center gap-5">
                <ScoreRing
                  value={Math.round(result.risk_score)}
                  tone={scoreTone(result.risk_score)}
                  size={116}
                  suffix="risk /100"
                />
                <div className="space-y-1.5">
                  <div
                    className={`flex items-center gap-2 font-mono text-[13px] font-600 ${
                      result.is_anomaly ? "text-signal" : "text-clearance"
                    }`}
                  >
                    {result.is_anomaly ? <ShieldAlert size={15} /> : <ShieldCheck size={15} />}
                    {result.is_anomaly ? "Anomaly flagged" : "Within baseline"}
                  </div>
                  <div className="font-mono text-[10.5px] text-ink-faint dark:text-paper/40">
                    raw IF score {result.raw_isolation_forest_score.toFixed(5)} · threshold{" "}
                    {result.operating_threshold?.toFixed(5)}
                  </div>
                  {result.event_id && (
                    <div className="font-mono text-[10.5px] text-ink-faint dark:text-paper/40">
                      {result.event_id}
                    </div>
                  )}
                </div>
              </div>

              <div className="border-t border-line pt-3 dark:border-line-dark">
                <div className="pb-2 font-mono text-[9.5px] uppercase tracking-widest2 text-ink-faint dark:text-paper/45">
                  Main reasons
                </div>
                <ul className="space-y-2">
                  {result.reasons.map((r) => (
                    <li
                      key={r.feature}
                      className="flex items-start gap-2.5 border border-line p-2.5 dark:border-line-dark"
                    >
                      <span
                        className={`mt-px flex-none font-mono text-[10px] font-600 tabular-nums ${riskTone(
                          r.magnitude * 10,
                        )}`}
                      >
                        ×{r.magnitude.toFixed(1)}
                      </span>
                      <span className="font-mono text-[10.5px] leading-relaxed text-ink-soft dark:text-paper/60">
                        {r.description}
                      </span>
                    </li>
                  ))}
                  {result.reasons.length === 0 && (
                    <li className="font-mono text-[10.5px] text-ink-faint dark:text-paper/40">
                      No significant deviations from the baseline.
                    </li>
                  )}
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center gap-2 font-mono text-[11px] text-ink-faint dark:text-paper/40">
              <BrainCircuit size={22} className="opacity-40" />
              Run an analysis to see the model&apos;s verdict.
            </div>
          )}
        </Panel>
      </div>

      {/* batch demo */}
      <Panel
        label="Batch simulation — 3 event types"
        right={
          <button
            onClick={() => analyze(null, "batch")}
            disabled={busy !== null}
            className="inline-flex items-center gap-1.5 border border-line px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wideish text-ink-soft transition-colors hover:border-signal hover:text-signal dark:border-line-dark dark:text-paper/60 disabled:opacity-50"
          >
            {busy === "batch" ? <Loader2 size={12} className="animate-spin" /> : <ArrowRight size={12} />}
            Run batch
          </button>
        }
      >
        {batch ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left">
              <thead>
                <tr className="border-b border-line dark:border-line-dark">
                  {["Event", "Verdict", "Risk", "Level", "Top reason"].map((h) => (
                    <th
                      key={h}
                      className="pb-2 font-mono text-[9.5px] uppercase tracking-widest2 text-ink-faint dark:text-paper/45"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line dark:divide-line-dark">
                {batch.map((r, i) => (
                  <tr
                    key={r.event_id ?? i}
                    className="transition-colors hover:bg-paper-dim/60 dark:hover:bg-void-surface/50"
                  >
                    <td className="px-1 py-3 font-mono text-[11px] text-ink dark:text-paper">
                      {r.event_id}
                    </td>
                    <td className="px-1 py-3">
                      <span
                        className={`font-mono text-[10.5px] font-600 ${
                          r.is_anomaly ? "text-signal" : "text-clearance"
                        }`}
                      >
                        {r.is_anomaly ? "ANOMALY" : "normal"}
                      </span>
                    </td>
                    <td
                      className={`px-1 py-3 font-mono text-[12px] font-600 tabular-nums ${riskTone(
                        r.risk_score,
                      )}`}
                    >
                      {r.risk_score.toFixed(1)}
                    </td>
                    <td className="px-1 py-3">
                      <SeverityBadge level={levelToSeverity[r.risk_level]} />
                    </td>
                    <td className="px-1 py-3 font-mono text-[10.5px] text-ink-soft dark:text-paper/60">
                      {r.reasons[0]?.feature ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="font-mono text-[11px] text-ink-faint dark:text-paper/40">
            Score {BATCH_EVENTS.length} NSL-KDD-style events at once (brute-force pattern, port
            scan, normal HTTP).
          </p>
        )}
      </Panel>
    </div>
  );
}
