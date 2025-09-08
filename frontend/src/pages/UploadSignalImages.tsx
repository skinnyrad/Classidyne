import React, { useEffect, useState } from "react";
import {
  Container,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  Grid,
  Button,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Stack,
} from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import CloseIcon from "@mui/icons-material/Close";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const UploadSignalImages: React.FC = () => {
  const [helpOpen, setHelpOpen] = useState(false);
  const [snackbar, setSnackbar] = useState<string | null>(null);
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const queryClient = useQueryClient();

  const fetchStats = async () => {
    const res = await fetch("/api/stats");
    const data = await res.json();
    if (!data.success)
      throw new Error(data.message ?? "Failed to fetch stats.");
    return {
      embedding_status: data.embedding_status,
      waterfall_size: data.waterfall_size,
      fft_size: data.fft_size,
    };
  };

  // React Query data fetching
  const {
    data: stats,
    error,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
    refetchInterval: 3000, // Every 3 seconds
    refetchOnWindowFocus: "always", // Always refetch when window gains focus
    refetchOnMount: "always", // Always refetch on mount
    retry: false, // Don't retry automatically on error (you can change this)
  });

  // Start embedding handler
  const startEmbedding = async () => {
    setEmbeddingLoading(true);
    try {
      const res = await fetch("/api/start-embedding", {
        method: "POST",
      });
      const data = await res.json();
      setSnackbar(
        data.success
          ? "Embedding process started!"
          : data.message || "Failed to start embedding."
      );
      refetch(); // refetch stats right after starting
    } catch (e: any) {
      setSnackbar(e.message ?? "Failed to start embedding.");
    } finally {
      setEmbeddingLoading(false);
    }
  };

  return (
    <Container sx={{ width: "70vw", my: 3 }}>
      <Typography variant="h4" gutterBottom>
        Embedding Dashboard
      </Typography>
      {/* Help Bar */}
      <Button
        sx={{
          width: "100%",
          mb: 2,
        }}
        onClick={() => setHelpOpen(true)}
        variant="contained"
      >
        <HelpOutlineIcon sx={{ mr: 1 }} />
        <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
          Upload Instructions
        </Typography>
      </Button>

      {/* Help Dialog */}
      <Dialog
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Adding New RF Signal Images
          <IconButton
            aria-label="close"
            onClick={() => setHelpOpen(false)}
            sx={{
              position: "absolute",
              right: 8,
              top: 8,
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            Follow these steps to add new images to the vector database:
          </Typography>

          <List>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    <strong>Organize your images:</strong> Place each image in
                    the correct folder based on its signal type.
                    <br />
                    Example: <code>datasets/waterfall/wifi/</code> for waterfall
                    WiFi signals, <code>datasets/fft/bluetooth/</code> for fft
                    Bluetooth signals.
                  </span>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    <strong>Folder structure:</strong> Ensure all images are in
                    folders following this pattern:
                    <br />
                    <code>datasets/waterfall/signal-type/</code> and{" "}
                    <code>datasets/fft/signal-type/</code>.<br />
                    Where <code>signal-type</code> is the specific type of
                    signal (e.g., wifi, bluetooth).
                  </span>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    <strong>Unknown signals:</strong> If you're unsure of the
                    signal type, place the image in{" "}
                    <code>datasets/image-type/unknown/</code>.
                  </span>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    <strong>Upload:</strong> Once you've organized your images,
                    click the 'Upload' button to add them to the signal
                    database.
                  </span>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    <strong>Creating New Signal Folders:</strong> You can create
                    new signal folders by navigating to the{" "}
                    <code>datasets</code> folder and creating a new folder with
                    the desired signal type.
                    <br />
                    For example, to create a new "Zigbee" folder for Zigbee
                    signals, navigate to <code>datasets</code> and create a
                    folder named <code>zigbee</code>.
                  </span>
                }
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary={
                  <span>
                    To reset the database, delete the{" "}
                    <code>image-vectordb.db</code> file before starting the
                    embedding process.
                  </span>
                }
              />
            </ListItem>
          </List>

          <Typography variant="body1" sx={{ mt: 2 }}>
            <strong>Note:</strong> All signal image folders must be located
            within the <code>datasets</code> directory.
          </Typography>

          <Typography variant="body2" sx={{ mt: 1 }}>
            <em>
              Supported image formats: jpg, jpeg, png, gif, tiff, tif, bmp, webp
            </em>
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHelpOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Main Panel */}
      {/* Main Panel */}
      <Paper elevation={3} sx={{ p: 4 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {(error as Error).message}
          </Alert>
        )}
        <Grid container spacing={4}>
          <Grid size={8}>
            <Stack spacing={2}>
              <Typography variant="h6">Embedding Status:</Typography>
              <Paper sx={{ p: 2 }}>
                <Typography variant="body1" sx={{ fontWeight: "bold" }}>
                  {stats?.embedding_status ?? "Loading..."}
                </Typography>
              </Paper>
              <Button
                variant="contained"
                onClick={startEmbedding}
                sx={{ width: "fit-content" }}
                disabled={
                  embeddingLoading ||
                  ["Running", "Starting"].includes(
                    stats?.embedding_status ?? ""
                  )
                }
              >
                {embeddingLoading ? "Starting..." : "Start Embedding"}
              </Button>
              <Typography variant="body2" color="text.secondary">
                If new images were added, trigger re-embedding here.
              </Typography>
            </Stack>
          </Grid>
          <Grid size={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Embedded Image Counts
              </Typography>
              <Stack spacing={1}>
                <Typography>
                  <b>Waterfall:</b> {stats?.waterfall_size ?? "--"}
                </Typography>
                <Typography>
                  <b>FFT:</b> {stats?.fft_size ?? "--"}
                </Typography>
              </Stack>
            </Paper>
          </Grid>
        </Grid>
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

export default UploadSignalImages;
