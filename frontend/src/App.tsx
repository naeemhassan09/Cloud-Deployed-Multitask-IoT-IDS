import React, { useState } from "react";
import { AppBar, Toolbar, Typography, Container, Grid, Box, Alert, Snackbar } from "@mui/material";
import UploadCard from "./components/UploadCard";
import PreviewCard from "./components/PreviewCard";
import PipelineStepper from "./components/PipelineStepper";
import Dashboard from "./components/Dashboard";
import { uploadCsvAndPredict } from "./api";
import type { PredictFileResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<PredictFileResponse | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; msg: string; sev: "success" | "error" }>({
    open: false,
    msg: "",
    sev: "success",
  });

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setData(null);
    try {
      const res = await uploadCsvAndPredict(file);
      setData(res);
      setSnack({ open: true, msg: "Pipeline completed successfully.", sev: "success" });
    } catch (e: any) {
      setSnack({ open: true, msg: e?.message ?? "Failed to run pipeline.", sev: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 500px at 20% 0%, rgba(59,130,246,0.20), transparent 55%), radial-gradient(900px 500px at 80% 10%, rgba(16,185,129,0.14), transparent 50%), #f6f7fb",
      }}
    >
      <AppBar position="sticky" elevation={0} sx={{ bgcolor: "rgba(255,255,255,0.85)", color: "inherit", backdropFilter: "blur(10px)" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Multitask IoT IDS – Demo
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Raw CSV → Preprocess → CNN → Predictions
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box mb={3}>
          <Typography variant="h4">Cloud-Deployed Multitask IDS Pipeline</Typography>
          <Typography variant="body1" color="text.secondary">
            Upload raw CIC IoT packet CSV. The backend validates required features, performs preprocessing, and runs multitask inference.
          </Typography>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <UploadCard
              file={file}
              setFile={setFile}
              loading={loading}
              onRun={run}
              onPreviewReady={setPreview}
            />
          </Grid>

          <Grid item xs={12} md={8}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <PipelineStepper stages={data?.stages} />
              </Grid>
              <Grid item xs={12}>
                <PreviewCard preview={preview} />
              </Grid>
            </Grid>
          </Grid>

          <Grid item xs={12}>
            {data ? (
              <Dashboard data={data} />
            ) : (
              <Alert severity="info">
                Upload a raw CSV and click “Run pipeline” to generate results.
              </Alert>
            )}
          </Grid>
        </Grid>
      </Container>

      <Snackbar open={snack.open} autoHideDuration={6000} onClose={() => setSnack((s) => ({ ...s, open: false }))}>
        <Alert severity={snack.sev} onClose={() => setSnack((s) => ({ ...s, open: false }))} sx={{ width: "100%" }}>
          {snack.msg}
        </Alert>
      </Snackbar>
    </Box>
  );
}