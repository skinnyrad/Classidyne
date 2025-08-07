import React, { useState } from "react";
import {
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
  CardContent,
  CardMedia,
  Stack,
  Alert,
  Paper,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { useEffect } from "react";

type Collection = "fft" | "waterfall";

type ClassScore = {
  class: string;
  confidence: number;
  frequency_range: string;
};

const SignalClassification: React.FC = () => {
  const [collection, setCollection] = useState<Collection>("waterfall");
  const [threshold, setThreshold] = useState<number>(0.6);
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null); // For preview
  const [classScores, setClassScores] = useState<ClassScore[]>([]);
  const [collage, setCollage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;
    setError(null);
    setClassScores([]);
    setCollage(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("query_image", file);
    formData.append("collection", collection);
    formData.append("similarity_threshold", threshold.toString());

    fetch("http://localhost:8000/api/classify", {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) throw new Error("Error from server.");
        return res.json();
      })
      .then((data) => {
        setClassScores(data.class_scores || []);
        setCollage(data.collage_image || null);
      })
      .catch(() => {
        setError("Failed to classify image.");
      })
      .finally(() => {
        setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
              <Stack direction={"row"} spacing={2} alignItems="center">
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
            {/* Display file preview if file is selected */}
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
      {loading && <LinearProgress />}
      {error && <Alert severity="error">{error}</Alert>}
      {!!classScores.length && (
        <Box mb={2}>
          <Typography variant="h6" gutterBottom>
            Class Scores
          </Typography>
          <Stack spacing={2}>
            {classScores.map((score) => (
              <Paper key={score.class} sx={{ p: 2 }}>
                <Stack direction={"row"} spacing={2} alignItems="center">
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
      {collage && (
        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            Simmilar Signals
          </Typography>
          <Card>
            <CardMedia
              component="img"
              src={`data:image/png;base64,${collage}`}
              alt="Collage"
            />
          </Card>
        </Box>
      )}
    </Container>
  );
};

export default SignalClassification;
