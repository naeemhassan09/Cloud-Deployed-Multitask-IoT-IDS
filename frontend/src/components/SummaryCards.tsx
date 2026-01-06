import React from 'react';
import { Grid, Card, CardContent, Typography, Chip, Stack } from '@mui/material';
import type { Summary, Metrics } from '../types';

interface Props {
  summary: Summary;
  metrics: Metrics;
}

const SummaryCards: React.FC<Props> = ({ summary, metrics }) => {
  const benignRate =
    summary.total_rows > 0 ? (summary.num_benign / summary.total_rows) * 100 : 0;
  const attackRate =
    summary.total_rows > 0 ? (summary.num_attacks / summary.total_rows) * 100 : 0;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="overline" color="text.secondary">
              Total Records
            </Typography>
            <Typography variant="h4" sx={{ mb: 1 }}>
              {summary.total_rows.toLocaleString()}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Chip
                label={`Benign: ${summary.num_benign.toLocaleString()} (${benignRate.toFixed(1)}%)`}
                color="success"
                size="small"
              />
              <Chip
                label={`Attacks: ${summary.num_attacks.toLocaleString()} (${attackRate.toFixed(1)}%)`}
                color="error"
                size="small"
              />
            </Stack>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="overline" color="text.secondary">
              Latency (ms)
            </Typography>
            <Typography variant="h5" sx={{ mb: 1 }}>
              {metrics.total_ms.toFixed(1)} ms total
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Preprocessing: {metrics.preprocessing_ms.toFixed(1)} ms
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Inference: {metrics.inference_ms.toFixed(1)} ms
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="overline" color="text.secondary">
              Pipeline Status
            </Typography>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Completed
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload CSV → Automatic feature preprocessing → Multitask CNN prediction → Results displayed in UI
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default SummaryCards;