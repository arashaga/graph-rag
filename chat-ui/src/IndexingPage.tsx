import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

type IndexingStatus = "pending" | "in_progress" | "completed" | "error" | null;

const API_BASE = "http://localhost:8000"; // Change if needed
const JOBID_STORAGE_KEY = "currentIndexingJobId";

export default function IndexingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [method, setMethod] = useState<"Standard" | "Fast">("Standard");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<IndexingStatus>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const [indexName, setIndexName] = useState("");

  // On mount: check for unfinished job in localStorage and resume polling
  useEffect(() => {
    const savedJobId = localStorage.getItem(JOBID_STORAGE_KEY);
    if (savedJobId) {
      setJobId(savedJobId);
      pollStatus(savedJobId);
    }
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
    // eslint-disable-next-line
  }, []);

  // Poll for job status
  const pollStatus = (id: string) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    setStatus("pending");
    pollingRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/status/${id}`);
        setStatus(res.data.status);
        if (res.data.status === "completed" || res.data.status === "error") {
          clearInterval(pollingRef.current!);
          localStorage.removeItem(JOBID_STORAGE_KEY);
          if (res.data.status === "error") setError(res.data.details || "Indexing failed.");
        }
      } catch {
        setError("Could not fetch job status.");
        clearInterval(pollingRef.current!);
        localStorage.removeItem(JOBID_STORAGE_KEY);
      }
    }, 2000);
  };

  // Start polling and store jobId on upload
  const handleUpload = async () => {
    if (!file) {
      setError("Please select a .txt file.");
      return;
    }
    setError(null);
    setStatus("pending");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("index_name", indexName);

    try {
      const res = await axios.post(
        `${API_BASE}/upload/?method=${method}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setJobId(res.data.job_id);
      setStatus(res.data.status);
      localStorage.setItem(JOBID_STORAGE_KEY, res.data.job_id);
      pollStatus(res.data.job_id);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Upload failed, check backend."
      );
      setStatus(null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (!selected.name.endsWith(".txt")) {
        setError("Only .txt files are allowed.");
        setFile(null);
        return;
      }
      setFile(selected);
      setError(null);
    }
  };

  // Reset only if no active job
  const reset = () => {
    setFile(null);
    setMethod("Standard");
    setJobId(null);
    setStatus(null);
    setError(null);
    if (pollingRef.current) clearInterval(pollingRef.current);
    localStorage.removeItem(JOBID_STORAGE_KEY);
  };

  // Disable upload/reset when a job is active
  const jobActive = jobId && status && status !== "completed" && status !== "error";

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      <div className="bg-white p-6 rounded-2xl shadow-md w-full max-w-md">
        <h2 className="text-xl font-bold mb-4">Index a .txt File</h2>

        <div className="mb-4">
          <label className="block mb-2 text-sm">Choose file</label>
          <input
            type="file"
            accept=".txt"
            ref={fileInputRef}
            onChange={handleFileSelect}
            disabled={jobActive}
            className={"block w-full border p-2 rounded" + (jobActive ? "opacity-50 cursor-not-allowed" : "")}
          />
          {file && (
            <span className="text-sm text-gray-600 mt-1 block">{file.name}</span>
          )}
        </div>

        <div className="mb-4">
          <label className="block mb-2 text-sm">Indexing Method</label>
          <select
            value={method}
            onChange={e => setMethod(e.target.value as "Standard" | "Fast")}
            className={"w-full border p-2 rounded"+ (jobActive ? "opacity-50 cursor-not-allowed" : "")}
            disabled={jobActive}

          >
            <option value="Standard">Standard (Default)</option>
            <option value="Fast">Fast</option>
          </select>
        </div>
        <div className="mb-4">
          <label className="block mb-2 text-sm">Index Name</label>
          <input
            type="text"
            value={indexName}
            onChange={e => setIndexName(e.target.value)}
            className={"w-full border p-2 rounded"+ (jobActive ? "opacity-50 cursor-not-allowed" : "")}
            disabled={jobActive}
            maxLength={32}
            placeholder="e.g. alice_index"
            required
          />
        </div>

        <button
          className={"w-full bg-blue-600 text-white font-bold py-2 px-4 rounded hover:bg-blue-700"+ (jobActive ? "opacity-50 cursor-not-allowed" : "")}
          onClick={handleUpload}
          disabled={!file || jobActive}
        >
          Start Indexing
        </button>

        <button
          className={"w-full mt-2 bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded hover:bg-gray-300"+ (jobActive ? "opacity-50 cursor-not-allowed" : "")}
          onClick={reset}
          disabled={jobActive}
        >
          Reset
        </button>

        {status && (
          <div className="mt-4">
            <div className="flex items-center space-x-2">
              <span className="font-semibold">Status:</span>
              <span
                className={
                  status === "completed"
                    ? "text-green-600"
                    : status === "error"
                    ? "text-red-600"
                    : "text-yellow-600"
                }
              >
                {status.charAt(0).toUpperCase() + status.slice(1).replace("_", " ")}
              </span>
              {status === "in_progress" && (
                <svg
                  className="animate-spin h-5 w-5 text-yellow-600"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                  />
                </svg>
              )}
            </div>
            {jobId && <div className="text-xs text-gray-400 mt-1">Job ID: {jobId}</div>}
          </div>
        )}

        {error && (
          <div className="mt-4 text-red-600 font-semibold">{error}</div>
        )}
      </div>
    </div>
  );
}
