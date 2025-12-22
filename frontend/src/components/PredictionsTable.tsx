import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Paper,
  Chip,
  Typography
} from '@mui/material';
import type { PredictionRow } from '../types';

interface Props {
  rows: PredictionRow[];
}

const PredictionsTable: React.FC<Props> = ({ rows }) => {
  return (
    <Card sx={{ mt: 3 }}>
      <CardHeader
        title="Detailed Predictions"
        subheader="Sample of row-level device and intrusion outputs"
      />
      <CardContent>
        {rows.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            Upload a CSV file to see detailed predictions.
          </Typography>
        ) : (
          <TableContainer component={Paper} sx={{ maxHeight: 420 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Device ID</TableCell>
                  <TableCell>Device Name</TableCell>
                  <TableCell>Attack Label</TableCell>
                  <TableCell>Attack Score</TableCell>
                  <TableCell>Device Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.slice(0, 200).map((row) => (
                  <TableRow key={row.row_index} hover>
                    <TableCell>{row.row_index}</TableCell>
                    <TableCell>{row.device_id}</TableCell>
                    <TableCell>{row.device_name}</TableCell>
                    <TableCell>
                      <Chip
                        label={row.attack_label}
                        color={row.attack_label.toLowerCase() === 'benign' ? 'success' : 'error'}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{row.attack_score.toFixed(3)}</TableCell>
                    <TableCell>{(row.device_confidence * 100).toFixed(1)}%</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
        {rows.length > 200 && (
          <Typography variant="caption" color="text.secondary">
            Showing first 200 rows only for UI responsiveness.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default PredictionsTable;