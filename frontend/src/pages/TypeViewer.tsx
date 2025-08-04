import React from 'react';
import { Typography, Container } from '@mui/material';

const TypeViewer: React.FC = () => (
  <Container sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>
      Type Viewer
    </Typography>
    <Typography>
      This is the Type Viewer page.
    </Typography>
  </Container>
);

export default TypeViewer;
