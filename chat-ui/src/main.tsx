import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import App from "./App";
import IndexingPage from "./IndexingPage";
import "./index.css";

// createRoot(document.getElementById('root')!).render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
// )


ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <nav className="p-4 flex gap-4 bg-blue-100">
        <Link to="/" className="text-blue-700 font-semibold">Chat</Link>
        <Link to="/indexing" className="text-blue-700 font-semibold">Indexing</Link>
      </nav>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/indexing" element={<IndexingPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);