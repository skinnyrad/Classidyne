import React from "react";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import SignalClassification from "./pages/SignalClassification";
import UploadSignalImages from "./pages/UploadSignalImages";
import ManageImages from "./pages/ManageImages";
import TypeViewer from "./pages/TypeViewer";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#d690cf",
    },
  },
  typography: { fontFamily: "Open Sans, Arial" },
  shape: {
    borderRadius: 15,
  },
});

const App: React.FC = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Routes>
          <Route
            path="/signal-classification"
            element={<SignalClassification />}
          />
          <Route
            path="/upload-signal-images"
            element={<UploadSignalImages />}
          />
          <Route path="/manage-images" element={<ManageImages />} />
          <Route path="/type-viewer" element={<TypeViewer />} />
          <Route path="*" element={<SignalClassification />} />
        </Routes>
      </Router>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
