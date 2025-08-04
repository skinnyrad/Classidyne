import React from 'react';
import { Typography, Container } from '@mui/material';

const SearchImage: React.FC = () => (
  <Container sx={{ mt: 4 }}>
    <Typography variant="h4" gutterBottom>
      Search Image
    </Typography>
    <Typography>
      This is the Search Image page.
    </Typography>
  </Container>
);

export default SearchImage;
