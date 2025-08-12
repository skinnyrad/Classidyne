import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Dialog,
  Container,
  Typography,
  Box,
  Button,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  LinearProgress,
  Grid,
  Card,
  CardMedia,
  Stack,
  Alert,
  Paper,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";

type Collection = "fft" | "waterfall";
type ClassScore = {
  class: string;
  confidence: number;
  frequency_range: string;
};

const classifyImage = async ({
  file,
  collection,
  threshold,
}: {
  file: File;
  collection: Collection;
  threshold: number;
}) => {
  const formData = new FormData();
  formData.append("query_image", file);
  formData.append("collection", collection);
  formData.append("similarity_threshold", threshold.toString());

  const res = await fetch("http://localhost:8000/api/classify", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Error from server.");
  return res.json();
};

const SignalClassification: React.FC = () => {
  const [collection, setCollection] = useState<Collection>("waterfall");
  const [threshold, setThreshold] = useState<number>(0.6);
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);

  // React Query mutation for classification
  const { mutate, data, error, isPending, reset } = useMutation({
    mutationFn: classifyImage,
  });

  // Auto-trigger mutation whenever file, collection, threshold change
  React.useEffect(() => {
    if (!file) {
      reset();
      return;
    }
    mutate({ file, collection, threshold });
    // eslint-disable-next-line
  }, [file, collection, threshold]);

  const handleCollection = (
    _: React.MouseEvent<HTMLElement>,
    val: Collection | null
  ) => {
    if (val) setCollection(val);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setFileUrl(URL.createObjectURL(e.target.files[0]));
    }
  };

  return (
    <Container sx={{ mt: 3, width: "70vw" }}>
      <Typography variant="h4" gutterBottom>
        Signal Classification
      </Typography>

      {/* Controls for upload, collection, threshold, live */}

      <Paper elevation={3} sx={{ p: 3, mb: 2 }}>
        <Grid container spacing={2}>
          <Grid size={6}>
            <Stack spacing={2}>
              <Button
                variant={file ? "contained" : "outlined"}
                component="label"
                startIcon={<UploadFileIcon />}
                sx={{ width: "100%", minHeight: "10vh" }}
              >
                {file ? file.name : "Upload Image"}
                <input
                  hidden
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </Button>
              <Stack direction="row" spacing={2} alignItems="center">
                <ToggleButtonGroup
                  color="primary"
                  value={collection}
                  exclusive
                  onChange={handleCollection}
                  aria-label="Collection Selector"
                >
                  <ToggleButton value="waterfall">Waterfall</ToggleButton>
                  <ToggleButton value="fft">FFT</ToggleButton>
                </ToggleButtonGroup>
                <Stack spacing={1}>
                  <Typography gutterBottom>
                    Similarity Threshold ({threshold * 100}%)
                  </Typography>
                  <Slider
                    min={0}
                    max={100}
                    step={10}
                    value={threshold * 100}
                    onChange={(_, value) =>
                      setThreshold((value as number) / 100)
                    }
                    valueLabelDisplay="auto"
                  />
                </Stack>
              </Stack>
            </Stack>
          </Grid>
          <Grid size={6}>
            {fileUrl && (
              <Box sx={{ display: "flex", justifyContent: "center" }}>
                <Card>
                  <CardMedia
                    component="img"
                    src={fileUrl}
                    alt="Uploaded file preview"
                  />
                </Card>
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>

      {/* Display results */}

      {isPending && <LinearProgress />}
      {error && <Alert severity="error">{String(error)}</Alert>}
      {!!data?.class_scores?.length && (
        <Box mb={2}>
          <Typography variant="h6" gutterBottom>
            Class Scores
          </Typography>
          <Stack spacing={2}>
            {data.class_scores.map((score: ClassScore) => (
              <Paper key={score.class} sx={{ p: 2 }}>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Typography variant="h4" sx={{ width: "10%" }}>
                    {score.confidence.toFixed(0)}%
                  </Typography>
                  <Typography variant="h6" sx={{ width: "15%" }}>
                    <b>{score.class.toUpperCase()}</b>
                  </Typography>
                  <Typography sx={{ maxWidth: "60%" }}>
                    Frequency range: {score.frequency_range}
                  </Typography>
                </Stack>
              </Paper>
            ))}
          </Stack>
        </Box>
      )}
      {data?.collage_image && (
        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            Simmilar Signals
          </Typography>
          <Card>
            <CardMedia
              component="img"
              src={`data:image/png;base64,${data.collage_image}`}
              alt="Collage"
            />
          </Card>
        </Box>
      )}
    </Container>
  );
};

export default SignalClassification;
