import React from 'react';
import { Typography, Container } from '@mui/material';

const DeleteImage: React.FC = () => (
  <Container sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>
      Delete Image
    </Typography>
    <Typography>
      This is the Delete Image page.
    </Typography>
  </Container>
);

export default DeleteImage;
