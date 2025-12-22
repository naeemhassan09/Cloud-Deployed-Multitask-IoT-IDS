import React, { useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Chip,
  TextField,
  Tooltip as MuiTooltip,
  Stack,
  Divider,
} from "@mui/material";
import type { PredictFileResponse, PredictionRow } from "../types";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Tooltip,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const ATTACK_COLORS: Record<string, string> = {
  benign: "#22c55e",
  dos: "#ef4444",
  "brute force": "#f97316",
  recon: "#a855f7",
  spoofing: "#eab308",
  "web-based": "#3b82f6",
  ddos: "#06b6d4",
  mirai: "#6366f1",
};

function pct(n: number, total: number) {
  return total > 0 ? (n / total) * 100 : 0;
}

function truncate(s: string, n = 20) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

function prettyDevice(s: string) {
  // shorten MAC-like identifiers
  if (s.includes(":") && s.length >= 14) return `${s.slice(0, 8)}…${s.slice(-5)}`;
  return s;
}

function toPieWithPct(dist: Record<string, number>) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(dist)
    .map(([name, value]) => ({ name, value, pct: (value / total) * 100 }))
    .sort((a, b) => b.value - a.value);
}

export default function Dashboard({ data }: { data: PredictFileResponse }) {
  const [q, setQ] = useState("");

  const totalRows = data.summary.total_rows || 0;
  const benignRate = totalRows > 0 ? (data.summary.num_benign / totalRows) * 100 : 0;
  const attackRate = 100 - benignRate;

  const attackPie = useMemo(
    () => toPieWithPct(data.summary.attack_distribution),
    [data.summary.attack_distribution]
  );

  const deviceBar = useMemo(() => {
    const arr = Object.entries(data.summary.device_distribution).map(([name, value]) => ({
      name,
      label: truncate(prettyDevice(name), 22),
      value,
    }));
    return arr.sort((a, b) => b.value - a.value).slice(0, 8);
  }, [data.summary.device_distribution]);

  const filtered: PredictionRow[] = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return data.predictions;
    return data.predictions.filter(
      (r) =>
        String(r.device_name).toLowerCase().includes(s) ||
        String(r.attack_label).toLowerCase().includes(s)
    );
  }, [q, data.predictions]);

  // Metrics formatting (handles older responses too)
  const m: any = data.metrics as any;
  const totalMs = m?.total_ms ?? 0;
  const preMs = m?.preprocessing_ms ?? 0;
  const infMs = m?.inference_ms ?? 0;

  const prePerRow = m?.per_row_preprocess_ms;
  const infPerRow = m?.per_row_inference_ms;
  const thr = m?.throughput_rows_per_sec;

  return (
    <Box>
      <Grid container spacing={2}>
        {/* KPI: total rows */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="overline" color="text.secondary">
                Total rows
              </Typography>
              <Typography variant="h4">{totalRows.toLocaleString()}</Typography>
              <Box mt={1} display="flex" gap={1} flexWrap="wrap">
                <Chip size="small" color="success" label={`Benign ${benignRate.toFixed(1)}%`} />
                <Chip size="small" color="error" label={`Attacks ${attackRate.toFixed(1)}%`} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* KPI: latency + per-row + throughput */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="overline" color="text.secondary">
                Latency
              </Typography>
              <Typography variant="h5">{Number(totalMs).toFixed(1)} ms</Typography>

              <Typography variant="body2" color="text.secondary">
                Preprocess: {Number(preMs).toFixed(1)} ms
                {prePerRow != null && ` (${Number(prePerRow).toFixed(3)} ms/row)`}
              </Typography>

              <Typography variant="body2" color="text.secondary">
                Inference: {Number(infMs).toFixed(1)} ms
                {infPerRow != null && ` (${Number(infPerRow).toFixed(3)} ms/row)`}
              </Typography>

              {thr != null && (
                <Typography variant="body2" color="text.secondary">
                  Throughput: {Number(thr).toFixed(0)} rows/s
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* KPI: model snapshot */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="overline" color="text.secondary">
                Model snapshot
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Multitask CNN backbone + dual heads
              </Typography>
              <Box mt={1} display="flex" gap={1} flexWrap="wrap">
                {data.eval_summary?.attack_accuracy != null && (
                  <Chip size="small" label={`Attack Acc: ${Number(data.eval_summary.attack_accuracy).toFixed(3)}`} />
                )}
                {data.eval_summary?.device_accuracy != null && (
                  <Chip size="small" label={`Device Acc: ${Number(data.eval_summary.device_accuracy).toFixed(3)}`} />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Attack distribution (fixed) */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: 420, display: "flex", flexDirection: "column" }}>
            <CardHeader title="Attack distribution" subheader="Predicted classes (batch)" />
            <CardContent sx={{ flex: 1, minHeight: 0 }}>
              <Box sx={{ height: 270 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={attackPie}
                      dataKey="value"
                      nameKey="name"
                      innerRadius="60%"
                      outerRadius="90%"
                      paddingAngle={2}
                      stroke="rgba(0,0,0,0.08)"
                    >
                      {attackPie.map((d) => (
                        <Cell key={d.name} fill={ATTACK_COLORS[d.name] ?? "#94a3b8"} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(v: any, _n: any, p: any) => [
                        `${v} (${p.payload.pct.toFixed(1)}%)`,
                        "Count",
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </Box>

              <Divider sx={{ my: 1 }} />

              {/* Clean legend (no overlap) */}
              <Stack direction="row" flexWrap="wrap" gap={1}>
                {attackPie.map((d) => (
                  <Box
                    key={d.name}
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      minWidth: 160,
                      py: 0.25,
                    }}
                  >
                    <Box
                      sx={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        bgcolor: ATTACK_COLORS[d.name] ?? "#94a3b8",
                      }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {d.name} — <b>{d.value}</b> ({d.pct.toFixed(1)}%)
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Top devices (fixed labels + tooltip) */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: 420, display: "flex", flexDirection: "column" }}>
            <CardHeader title="Top devices" subheader="Most frequent predicted devices" />
            <CardContent sx={{ flex: 1, minHeight: 0 }}>
              <Box sx={{ height: 330 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={deviceBar} layout="vertical" margin={{ left: 40, right: 20, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="label" width={170} />
                    <Tooltip
                      formatter={(v: any, _n: any, p: any) => [v, p.payload.name]}
                      labelFormatter={() => ""}
                    />
                    <Bar dataKey="value" radius={[6, 6, 6, 6]} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Table */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Row-level predictions"
              subheader="Search by device or attack label"
              action={
                <TextField
                  size="small"
                  placeholder="Search (e.g., spoofing, Echo Spot)"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                />
              }
            />
            <CardContent>
              <Box sx={{ overflow: "auto", borderRadius: 2, border: "1px solid rgba(15,23,42,0.08)" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: "rgba(15,23,42,0.03)" }}>
                      {["#", "Device", "Attack", "Attack conf", "Device conf"].map((h) => (
                        <th key={h} style={{ textAlign: "left", padding: "10px", fontSize: 13 }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.slice(0, 200).map((r) => {
                      const isBenign = String(r.attack_label).toLowerCase() === "benign";
                      const chipBg = isBenign ? "rgba(34,197,94,0.10)" : "rgba(239,68,68,0.10)";

                      return (
                        <tr key={r.row_index} style={{ borderTop: "1px solid rgba(15,23,42,0.06)" }}>
                          <td style={{ padding: "10px", fontSize: 13 }}>{r.row_index}</td>

                          <td style={{ padding: "10px", fontSize: 13 }}>
                            <MuiTooltip title={String(r.device_name)} placement="top" arrow>
                              <span>{prettyDevice(String(r.device_name))}</span>
                            </MuiTooltip>
                          </td>

                          <td style={{ padding: "10px", fontSize: 13 }}>
                            <span
                              style={{
                                padding: "4px 10px",
                                borderRadius: 999,
                                border: "1px solid rgba(15,23,42,0.10)",
                                background: chipBg,
                                textTransform: "lowercase",
                              }}
                            >
                              {r.attack_label}
                            </span>
                          </td>

                          {/* Use consistent units: show BOTH as percentages for demo */}
                          <td style={{ padding: "10px", fontSize: 13 }}>{(r.attack_score * 100).toFixed(1)}%</td>
                          <td style={{ padding: "10px", fontSize: 13 }}>{(r.device_confidence * 100).toFixed(1)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Box>

              <Typography variant="caption" color="text.secondary">
                Showing first 200 rows (filtered) for UI responsiveness.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}