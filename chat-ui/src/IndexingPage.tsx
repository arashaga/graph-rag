import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import styles from "./IndexingPage.module.css";

type IndexingStatus = "pending" | "in_progress" | "completed" | "error" | null;

const API_BASE = "http://localhost:50505"; // Updated to match backend port
const JOBID_STORAGE_KEY = "currentIndexingJobId";

export default function IndexingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [method, setMethod] = useState<"Standard" | "Fast">("Standard");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<IndexingStatus>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingRef = useRef<number | null>(null);
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
    pollingRef.current = window.setInterval(async () => {
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
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Disable upload/reset when a job is active
  const jobActive = Boolean(jobId && status && status !== "completed" && status !== "error");

  const getStatusIcon = () => {
    switch (status) {
      case "completed":
        return (
          <svg className={styles.statusIcon} viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="#10b981" />
            <path d="m9 12 2 2 4-4" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        );
      case "error":
        return (
          <svg className={styles.statusIcon} viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="#ef4444" />
            <path d="m15 9-6 6m0-6 6 6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        );
      case "in_progress":
      case "pending":
        return (
          <svg className={`${styles.statusIcon} ${styles.spinner}`} viewBox="0 0 24 24">
            <circle className={styles.spinnerTrack} cx="12" cy="12" r="10" stroke="#e5e7eb" strokeWidth="2" fill="none" />
            <circle className={styles.spinnerPath} cx="12" cy="12" r="10" stroke="#3b82f6" strokeWidth="2" fill="none" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className={styles.pageContainer}>
      <div className={styles.cardContainer}>
        <div className={styles.header}>
          <h1 className={styles.title}>Document Indexing</h1>
          <p className={styles.subtitle}>Upload and index your .txt files for AI-powered search</p>
        </div>

        <div className={styles.formContainer}>
          {/* File Upload Section */}
          <div className={styles.inputGroup}>
            <label className={styles.label}>Choose Document</label>
            <div className={styles.fileInputContainer}>
              <input
                type="file"
                accept=".txt"
                ref={fileInputRef}
                onChange={handleFileSelect}
                disabled={jobActive}
                className={styles.fileInput}
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className={`${styles.fileInputLabel} ${jobActive ? styles.disabled : ''}`}
              >
                <svg className={styles.uploadIcon} viewBox="0 0 24 24" fill="none">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {file ? file.name : 'Choose .txt file'}
              </label>
            </div>
          </div>

          {/* Index Name */}
          <div className={styles.inputGroup}>
            <label className={styles.label} htmlFor="index-name">Index Name</label>
            <input
              id="index-name"
              type="text"
              value={indexName}
              onChange={e => setIndexName(e.target.value)}
              className={`${styles.textInput} ${jobActive ? styles.disabled : ''}`}
              disabled={jobActive}
              maxLength={32}
              placeholder="e.g. my_document_index"
              required
            />
          </div>

          {/* Method Selection */}
          <div className={styles.inputGroup}>
            <label className={styles.label} htmlFor="method-select">Indexing Method</label>
            <select
              id="method-select"
              value={method}
              onChange={e => setMethod(e.target.value as "Standard" | "Fast")}
              className={`${styles.selectInput} ${jobActive ? styles.disabled : ''}`}
              disabled={jobActive}
            >
              <option value="Standard">Standard (Recommended)</option>
              <option value="Fast">Fast (Quick processing)</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className={styles.buttonGroup}>
            <button
              className={`${styles.primaryButton} ${(!file || jobActive) ? styles.disabled : ''}`}
              onClick={handleUpload}
              disabled={!file || jobActive}
            >
              {status === "in_progress" ? "Processing..." : "Start Indexing"}
            </button>
            <button
              className={`${styles.secondaryButton} ${jobActive ? styles.disabled : ''}`}
              onClick={reset}
              disabled={jobActive}
            >
              Reset
            </button>
          </div>

          {/* Status Display */}
          {status && (
            <div className={styles.statusContainer}>
              <div className={styles.statusHeader}>
                {getStatusIcon()}
                <div className={styles.statusInfo}>
                  <span className={styles.statusLabel}>Status</span>
                  <span className={`${styles.statusText} ${styles[status]}`}>
                    {status.charAt(0).toUpperCase() + status.slice(1).replace("_", " ")}
                  </span>
                </div>
              </div>
              {jobId && (
                <div className={styles.jobId}>
                  <span className={styles.jobIdLabel}>Job ID:</span>
                  <code className={styles.jobIdValue}>{jobId}</code>
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className={styles.errorContainer}>
              <svg className={styles.errorIcon} viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="#ef4444" strokeWidth="2" />
                <line x1="15" y1="9" x2="9" y2="15" stroke="#ef4444" strokeWidth="2" />
                <line x1="9" y1="9" x2="15" y2="15" stroke="#ef4444" strokeWidth="2" />
              </svg>
              <span className={styles.errorText}>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
