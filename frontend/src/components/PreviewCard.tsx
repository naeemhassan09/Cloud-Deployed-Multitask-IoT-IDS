import React from "react";
import { Card, CardContent, CardHeader, Typography, Box, Table, TableHead, TableRow, TableCell, TableBody } from "@mui/material";

export default function PreviewCard({
  preview,
}: {
  preview: { headers: string[]; rows: any[]; rowCountGuess: number } | null;
}) {
  return (
    <Card>
      <CardHeader title="CSV preview" subheader="First 15 rows parsed in browser" />
      <CardContent>
        {!preview ? (
          <Typography variant="body2" color="text.secondary">
            Select a CSV to see a preview.
          </Typography>
        ) : (
          <>
            <Box display="flex" gap={2} flexWrap="wrap" mb={2}>
              <Typography variant="body2">
                Columns: <b>{preview.headers.length}</b>
              </Typography>
              <Typography variant="body2">
                Preview rows: <b>{preview.rows.length}</b>
              </Typography>
            </Box>

            <Box sx={{ overflow: "auto", borderRadius: 2, border: "1px solid rgba(15,23,42,0.08)" }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    {preview.headers.slice(0, 10).map((h) => (
                      <TableCell key={h}><b>{h}</b></TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.rows.slice(0, 8).map((r, i) => (
                    <TableRow key={i} hover>
                      {preview.headers.slice(0, 10).map((h) => (
                        <TableCell key={h}>{String(r?.[h] ?? "")}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>

            <Typography variant="caption" color="text.secondary">
              Showing first 10 columns and 8 rows for readability.
            </Typography>
          </>
        )}
      </CardContent>
    </Card>
  );
}