import React from "react";
import { Card, CardContent, Typography, Stepper, Step, StepLabel, Box, Chip } from "@mui/material";
import type { Stages } from "../types";

const steps = [
  { key: "upload", label: "Upload" },
  { key: "validate", label: "Validate" },
  { key: "preprocess", label: "Preprocess" },
  { key: "predict", label: "Predict" },
  { key: "results", label: "Results" },
] as const;

function statusToIndex(stages?: Stages) {
  if (!stages) return 0;
  const order: (keyof Stages)[] = ["upload", "validate", "preprocess", "predict", "results"];
  let idx = 0;
  for (let i = 0; i < order.length; i++) {
    const s = stages[order[i]];
    if (s?.status === "done") idx = i + 1;
    if (s?.status === "error") return i;
  }
  return Math.min(idx, order.length);
}

export default function PipelineStepper({ stages }: { stages?: Stages }) {
  const active = statusToIndex(stages);
  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={2}>
          <Typography variant="h6">Pipeline Progress</Typography>
          {stages ? (
            <Chip
              size="small"
              label={`Preprocess ${stages.preprocess.ms?.toFixed(1) ?? "-"} ms • Inference ${stages.predict.ms?.toFixed(1) ?? "-"} ms`}
            />
          ) : (
            <Chip size="small" label="Awaiting upload" />
          )}
        </Box>

        <Stepper activeStep={Math.min(active, steps.length)} alternativeLabel>
          {steps.map((s) => (
            <Step key={s.key}>
              <StepLabel>{s.label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {stages && (
          <Box mt={2} display="flex" gap={1} flexWrap="wrap">
            <Chip size="small" label={`Rows: ${stages.upload.rows ?? "-"}`} />
            <Chip
              size="small"
              label={`Features: ${stages.validate.features_present ?? "-"} / ${stages.validate.features_expected ?? "-"}`}
            />
            <Chip size="small" label={`Used rows: ${stages.preprocess.used_rows ?? "-"}`} />
          </Box>
        )}
      </CardContent>
    </Card>
  );
}