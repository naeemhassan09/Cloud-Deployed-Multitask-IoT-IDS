import React, { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, Typography, Box, Button, LinearProgress, Divider } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import Papa from "papaparse";

type Preview = { headers: string[]; rows: any[]; rowCountGuess: number };

export default function UploadCard({
  file,
  setFile,
  loading,
  onRun,
  onPreviewReady,
}: {
  file: File | null;
  setFile: (f: File | null) => void;
  loading: boolean;
  onRun: () => void;
  onPreviewReady: (p: Preview | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);

  const prettySize = useMemo(() => {
    if (!file) return "";
    const mb = file.size / (1024 * 1024);
    return mb < 1 ? `${(file.size / 1024).toFixed(1)} KB` : `${mb.toFixed(2)} MB`;
  }, [file]);

  const parsePreview = (f: File) => {
    Papa.parse(f, {
      header: true,
      skipEmptyLines: true,
      preview: 15,
      complete: (res) => {
        const headers = (res.meta.fields ?? []).filter(Boolean) as string[];
        const rows = (res.data ?? []) as any[];
        onPreviewReady({ headers, rows, rowCountGuess: rows.length });
      },
      error: () => onPreviewReady(null),
    });
  };

  const selectFile = (f: File) => {
    setFile(f);
    parsePreview(f);
  };

  return (
    <Card sx={{ height: "100%" }}>
      <CardHeader
        title="Upload raw CSV"
        subheader="Raw CIC packet/flow CSV → backend preprocess → model inference → dashboard"
      />
      <CardContent>
        <Box
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            const f = e.dataTransfer.files?.[0];
            if (f) selectFile(f);
          }}
          onClick={() => inputRef.current?.click()}
          sx={{
            border: "2px dashed",
            borderColor: drag ? "primary.main" : "rgba(15,23,42,0.20)",
            borderRadius: 3,
            p: 3,
            cursor: "pointer",
            bgcolor: drag ? "rgba(59,130,246,0.06)" : "transparent",
            transition: "all 120ms ease",
          }}
        >
          <Box display="flex" alignItems="center" gap={2}>
            <CloudUploadIcon fontSize="large" />
            <Box>
              <Typography variant="subtitle1" fontWeight={800}>
                Drag & drop CSV here
              </Typography>
              <Typography variant="body2" color="text.secondary">
                or click to browse. Raw columns are allowed; required numeric features will be selected server-side.
              </Typography>
            </Box>
          </Box>
          {file && (
            <Box mt={2}>
              <Typography variant="body2">
                Selected: <b>{file.name}</b> • {prettySize}
              </Typography>
            </Box>
          )}
        </Box>

        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) selectFile(f);
          }}
        />

        <Divider sx={{ my: 2 }} />

        <Button
          variant="contained"
          fullWidth
          disabled={!file || loading}
          onClick={onRun}
        >
          Run pipeline
        </Button>

        {loading && (
          <Box mt={2}>
            <Typography variant="body2" color="text.secondary" mb={1}>
              Uploading and running inference…
            </Typography>
            <LinearProgress />
          </Box>
        )}
      </CardContent>
    </Card>
  );
}