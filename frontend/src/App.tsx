import React from "react";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import SignalClassification from "./pages/SignalClassification";
import UploadSignalImages from "./pages/UploadSignalImages";
import DeleteImage from "./pages/DeleteImage";
import SearchImage from "./pages/SearchImage";
import TypeViewer from "./pages/TypeViewer";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#d690cf",
    },
  },
});

const App: React.FC = () => (
  <ThemeProvider theme={darkTheme}>
    <CssBaseline />
    <Router>
      <Navbar />
      <Routes>
        <Route
          path="/signal-classification"
          element={<SignalClassification />}
        />
        <Route path="/upload-signal-images" element={<UploadSignalImages />} />
        <Route path="/delete-image" element={<DeleteImage />} />
        <Route path="/search-image" element={<SearchImage />} />
        <Route path="/type-viewer" element={<TypeViewer />} />
        <Route path="*" element={<SignalClassification />} />
      </Routes>
    </Router>
  </ThemeProvider>
);

export default App;
