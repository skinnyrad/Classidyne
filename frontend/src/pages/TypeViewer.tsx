import React, { useState } from "react";
import {
  Container,
  Typography,
  Paper,
  Box,
  Snackbar,
  Alert,
  LinearProgress,
  Stack,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  ToggleButtonGroup,
  ToggleButton,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";

const fetchTypes = async (collection: string) => {
  const endpoint =
    collection === "waterfall" ? "/api/waterfall_types" : "/api/fft_types";
  const res = await fetch(endpoint);
  const data = await res.json();
  if (!data.success) throw new Error(data.message ?? "Failed to fetch types");
  return data.types;
};

const fetchCollage = async (collection: string, type: string) => {
  const res = await fetch(
    `/api/type_collage?collection=${collection}&type=${encodeURIComponent(
      type
    )}`
  );
  const data = await res.json();
  if (!data.success) throw new Error(data.message ?? "Failed to fetch collage");
  return data.collage_base64;
};

const TypeViewer: React.FC = () => {
  const [collection, setCollection] = useState<"waterfall" | "fft">(
    "waterfall"
  );
  const [selectedType, setSelectedType] = useState("");
  const [snackbar, setSnackbar] = useState<string | null>(null);

  // Fetch types when collection changes
  const {
    data: types,
    error: typesError,
    isFetching: isFetchingTypes,
  } = useQuery({
    queryKey: ["types", collection],
    queryFn: () => fetchTypes(collection),
    refetchOnWindowFocus: false,
  });

  // Fetch collage only if type is picked
  const {
    data: collageBase64,
    error: collageError,
    isFetching: isFetchingCollage,
  } = useQuery({
    queryKey: ["collage", collection, selectedType],
    queryFn: () => fetchCollage(collection, selectedType),
    enabled: !!selectedType,
    refetchOnWindowFocus: false,
  });

  return (
    <Container sx={{ width: "70vw", my: 3 }}>
      <Typography variant="h4" gutterBottom>
        Signal Type Collage Browser
      </Typography>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Stack spacing={3}>
          {/* Control Row: Segmented and Dropdown */}
          <Box sx={{ display: "flex", gap: 3, alignItems: "center" }}>
            <ToggleButtonGroup
              value={collection}
              color="primary"
              exclusive
              onChange={(e, value) =>
                value && (setCollection(value), setSelectedType(""))
              }
              aria-label="Collection"
            >
              <ToggleButton value="waterfall">Waterfall</ToggleButton>
              <ToggleButton value="fft">FFT</ToggleButton>
            </ToggleButtonGroup>

            <FormControl sx={{ width: "100%" }}>
              <InputLabel>Signal Type</InputLabel>
              <Select
                label="Signal Type"
                color="primary"
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                disabled={!types || isFetchingTypes}
              >
                {types &&
                  types.map((type: string) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>
          </Box>

          {/* Loading indicator for types */}
          {isFetchingTypes && <LinearProgress sx={{ width: "100%" }} />}

          {/* Error display for types */}
          {typesError && (
            <Alert severity="error">{(typesError as Error).message}</Alert>
          )}

          {/* Collage output */}
          {selectedType && (
            <>
              {isFetchingCollage ? (
                <LinearProgress sx={{ width: "100%" }} />
              ) : collageError ? (
                <Alert severity="error">
                  {(collageError as Error).message}
                </Alert>
              ) : collageBase64 ? (
                <Box sx={{ textAlign: "center", mt: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    {collection.toUpperCase()} Signal Type: {selectedType}
                  </Typography>
                  <img
                    src={`data:image/png;base64,${collageBase64}`}
                    alt={`${selectedType} collage`}
                  />
                </Box>
              ) : null}
            </>
          )}
        </Stack>
      </Paper>
      <Snackbar
        open={!!snackbar}
        autoHideDuration={4000}
        onClose={() => setSnackbar(null)}
      >
        <Alert severity="info" sx={{ width: "100%" }}>
          {snackbar}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default TypeViewer;
