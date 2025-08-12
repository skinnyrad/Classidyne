import React, { useState } from "react";
import {
  Container,
  Typography,
  Paper,
  Box,
  Button,
  Snackbar,
  Alert,
  TextField,
  LinearProgress,
  Stack,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";

const findImage = async (identifier: string) => {
  const res = await fetch(
    `http://localhost:8000/api/find_image?identifier=${encodeURIComponent(
      identifier
    )}`
  );
  const data = await res.json();
  if (!data.success) throw new Error(data.message ?? "Not found");
  return data;
};

const deleteImage = async (identifier: string) => {
  const res = await fetch(
    `http://localhost:8000/api/delete_image?identifier=${encodeURIComponent(
      identifier
    )}`,
    {
      method: "DELETE",
    }
  );
  const data = await res.json();
  if (!data.success) throw new Error(data.message ?? "Delete failed");
  return data;
};

const ManageImages: React.FC = () => {
  const [input, setInput] = useState("");
  const [searchId, setSearchId] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<string | null>(null);

  // RUN SEARCH QUERY
  const { data, error, isFetching } = useQuery({
    queryKey: ["find_image", searchId],
    queryFn: () => findImage(searchId || ""),
    enabled: !!searchId,
    retry: false,
  });

  // DELETION MUTATION
  const mutation = useMutation({
    mutationFn: deleteImage,
    onSuccess: (data) => {
      setSnackbar(data.message);
      setSearchId(null); // Clear display after delete
    },
    onError: (err: any) => setSnackbar(err.message ?? "Delete failed"),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) setSearchId(input.trim());
  };

  const handleDelete = () => {
    if (searchId) mutation.mutate(searchId);
  };

  return (
    <Container sx={{ width: "70vw", my: 3 }}>
      <Typography variant="h4" gutterBottom>
        Search / Delete Signal Image
      </Typography>
      <Paper elevation={3} sx={{ p: 3 }}>
        {/* Search Bar */}
        <form onSubmit={handleSubmit}>
          <Stack direction="row" spacing={2}>
            <TextField
              label="File Hash or Filename"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              sx={{ flex: 1 }}
              placeholder="Type hash or filename…"
              autoFocus
            />
            <Button
              variant="contained"
              type="submit"
              disabled={!input || isFetching}
              sx={{ height: 56 }}
            >
              Search
            </Button>
          </Stack>
        </form>

        {isFetching && <LinearProgress sx={{ mt: 2 }} />}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {(error as Error).message}
          </Alert>
        )}

        {data && (
          <Stack spacing={2} sx={{ mt: 4 }}>
            <Paper sx={{ p: 2, wordBreak: "break-all" }}>
              <Typography variant="h6" gutterBottom>
                Image Information
              </Typography>
              <Typography>
                <b>File Hash:</b> {data.filehash}
              </Typography>
              <Typography>
                <b>Signal Class:</b> {data.class}
              </Typography>
              <Typography>
                <b>Filepath:</b> {data.filepath}
              </Typography>
            </Paper>
            <Paper sx={{ p: 2, textAlign: "center" }}>
              <Typography variant="subtitle2" gutterBottom>
                Image Preview
              </Typography>
              {data.image ? (
                <img
                  src={`data:image/png;base64,${data.image}`}
                  alt="Found"
                  style={{
                    width: 150,
                    height: 150,
                    objectFit: "cover",
                    borderRadius: 8,
                    border: "1px solid #ccc",
                  }}
                />
              ) : (
                <Typography color="text.secondary">
                  No Preview Available
                </Typography>
              )}
            </Paper>
            <Box>
              <Button
                color="error"
                variant="contained"
                onClick={handleDelete}
                disabled={mutation.isPending}
              >
                {mutation.isPending ? "Deleting..." : "Delete Image"}
              </Button>
            </Box>
          </Stack>
        )}
      </Paper>

      <Snackbar
        open={!!snackbar}
        autoHideDuration={4000}
        onClose={() => setSnackbar(null)}
      >
        <Alert
          severity={mutation.isError ? "error" : "info"}
          sx={{ width: "100%" }}
        >
          {snackbar}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default ManageImages;
