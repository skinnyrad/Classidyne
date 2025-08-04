import React from 'react';
import { Typography, Container } from '@mui/material';

const SignalClassification: React.FC = () => (
  <Container sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>
      Signal Classification
    </Typography>
    <Typography>
      This is the Signal Classification page.
    </Typography>
  </Container>
);

export default SignalClassification;
