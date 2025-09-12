import React from "react";
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material";
import { Link as RouterLink, useLocation } from "react-router-dom";

const tabs = [
  { label: "Signal Classification", path: "/signal-classification" },
  { label: "Upload Signal Images", path: "/upload-signal-images" },
  { label: "Manage Images", path: "/manage-images" },
  { label: "Type Viewer", path: "/type-viewer" },
];

const Navbar: React.FC = () => {
  const location = useLocation();

  return (
    <AppBar position="static">
      <Toolbar>
        <Box sx={{ display: "flex", alignItems: "center", mr: 2 }}>
          <img
            src="/SkinnyRD.png"
            alt="SkinnyRD Logo"
            style={{ height: 40, marginRight: 2 }}
          />
        </Box>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Classidyne
        </Typography>
        <Box>
          {tabs.map((tab) => (
            <Button
              key={tab.path}
              component={RouterLink}
              to={tab.path}
              color={location.pathname === tab.path ? "primary" : "inherit"}
              sx={{ ml: 1 }}
            >
              {tab.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
