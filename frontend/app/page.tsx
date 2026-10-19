"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  FileText,
  UploadCloud,
  Send,
  PlusCircle,
  Copy,
  Check,
  Bot,
  User,
  Sparkles,
  ShieldCheck,
  Zap,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  HelpCircle,
  FileCheck,
  X,
  MessageSquare,
  BarChart3,
  Network
} from "lucide-react";
import Link from "next/link";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  source?: string;
  requires_permission?: boolean;
  conversation_state?: string;
  metrics?: {
    chromadb_retrieval_time?: number;
    first_token_time?: number;
    llm_generation_time?: number;
    total_api_time?: number;
  };
  isStreaming?: boolean;
}

export default function Home() {
  // Input states
  const [question, setQuestion] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStage, setUploadStage] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [loading, setLoading] = useState(false);

  // Backend state
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedFileSize, setUploadedFileSize] = useState<string | null>(null);

  // Sync state with sessionStorage so it persists across pages
  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedId = sessionStorage.getItem("cognivault_documentId");
      if (storedId && !documentId) {
        setDocumentId(storedId);
        setUploadedFileName(sessionStorage.getItem("cognivault_fileName") || null);
        setUploadedFileSize(sessionStorage.getItem("cognivault_fileSize") || null);
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      if (documentId) {
        sessionStorage.setItem("cognivault_documentId", documentId);
        sessionStorage.setItem("cognivault_fileName", uploadedFileName || "");
        sessionStorage.setItem("cognivault_fileSize", uploadedFileSize || "");
      } else {
        sessionStorage.removeItem("cognivault_documentId");
        sessionStorage.removeItem("cognivault_fileName");
        sessionStorage.removeItem("cognivault_fileSize");
      }
    }
  }, [documentId, uploadedFileName, uploadedFileSize]);

  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");

  // UI layout states
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [dragActive, setDragActive] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Backend Health Ping Check
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/", { method: "GET" });
        if (res.ok) {
          setBackendStatus("online");
        } else {
          setBackendStatus("offline");
        }
      } catch (e) {
        setBackendStatus("offline");
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  // Format File Size
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  // Drag and Drop handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        setSelectedFile(file);
        setUploadMessage("");
      } else {
        setUploadMessage("Please select a PDF file (.pdf only).");
      }
    }
  };

  // File Select Handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        setSelectedFile(file);
        setUploadMessage("");
      } else {
        setUploadMessage("Please select a valid PDF file.");
        setSelectedFile(null);
      }
    }
  };

  // ==================================================
  // PDF UPLOAD (Stores exact document_id)
  // ==================================================
  
  const uploadPDF = async () => {
    if (!selectedFile) {
      setUploadMessage("Please select a PDF file first.");
      return;
    }

    setDocumentId(null);
    setUploadedFileName(null);
    setUploadedFileSize(null);
    setUploading(true);
    setUploadProgress(0);
    setUploadStage("Queued");
    setUploadMessage("");

    try {
      console.log("[UPLOAD] Starting upload for", selectedFile.name);
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://127.0.0.1:8000/api/upload-pdf", {
        method: "POST",
        body: formData
      });
      
      console.log("[UPLOAD] response status:", response.status);

      let data;
      try {
        data = await response.json();
      } catch (jsonErr) {
        throw new Error("Invalid response from server. Backend may have failed or returned a non-JSON page.");
      }

      if (!response.ok) {
        throw new Error(data.detail || data.message || "Upload failed with status " + response.status);
      }
      
      if (!data.task_id) {
        throw new Error("Backend did not return a valid task_id.");
      }
      
      const taskId = data.task_id;
      
      // Polling loop
      while (true) {
          await new Promise(resolve => setTimeout(resolve, 1500));
          
          const statusRes = await fetch(`http://127.0.0.1:8000/api/upload-status/${taskId}`);
          if (!statusRes.ok) {
              if (statusRes.status === 404) continue;
              throw new Error("Failed to fetch upload status.");
          }
          
          const statusData = await statusRes.json();
          console.log("[UPLOAD] Poll status:", statusData);
          
          if (statusData.stage) setUploadStage(statusData.stage);
          if (statusData.progress) setUploadProgress(statusData.progress);
          
          if (statusData.status === "completed") {
              const resData = statusData.result;
              setDocumentId(resData.document_id);
              setUploadedFileName(resData.filename);
              setUploadedFileSize(formatBytes(selectedFile.size));
              setUploadMessage(resData.message || "PDF indexed successfully.");
              setBackendStatus("online");

              if (messages.length === 0) {
                setMessages([
                  {
                    id: Date.now().toString(),
                    role: "assistant",
                    content: `Document **${resData.filename}** has been successfully parsed and indexed in Cognivault's ChromaDB vector store. Ask any question below to extract grounded answers!`,
                    timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                  }
                ]);
              }
              break;
          } else if (statusData.status === "failed") {
              throw new Error(statusData.error || "Background processing failed.");
          }
      }

    } catch (error: any) {
      console.error("UPLOAD ERROR:", error);
      setUploadMessage("Unable to upload PDF: " + (error.message || "Ensure the backend is running at http://127.0.0.1:8000."));
      setBackendStatus("offline");
    } finally {
      console.log("[UPLOAD] Resetting uploading state");
      setUploading(false);
      setUploadProgress(0);
      setUploadStage("");
    }
  };


  // Clear current document
  const clearDocument = () => {
    setDocumentId(null);
    setUploadedFileName(null);
    setUploadedFileSize(null);
    setSelectedFile(null);
    setUploadMessage("");
  };

  // Start New Conversation
  const startNewChat = () => {
    setMessages([]);
    setQuestion("");
  };

  // Copy AI Response to Clipboard
  const copyToClipboard = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMessageId(id);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  // ==================================================
  // ASK COGNIVAULT (Preserves exact NDJSON stream contract)
  // ==================================================
  const askCognivault = async (
    overridePrompt?: string,
    allowGeneralOverride?: boolean,
    permissionResponseOverride?: string,
    isPermissionChoice?: boolean
  ) => {
    let promptToSend = overridePrompt || question;

    if (isPermissionChoice) {
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      if (lastUserMsg) {
        promptToSend = lastUserMsg.content;
      }
    }

    if (!promptToSend.trim()) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (isPermissionChoice) {
      setMessages((prev) =>
        prev.map((m) => (m.requires_permission ? { ...m, requires_permission: false } : m))
      );
    }

    let currentMessages = messages;
    let assistantMessageId = Date.now().toString();

    if (!isPermissionChoice) {
      const userMessageId = Date.now().toString();
      assistantMessageId = (Date.now() + 1).toString();

      const userMsg: Message = {
        id: userMessageId,
        role: "user",
        content: promptToSend.trim(),
        timestamp
      };

      const assistantMsg: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp,
        isStreaming: true
      };

      currentMessages = [...messages, userMsg];
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
    } else {
      const assistantMsg: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp,
        isStreaming: true
      };
      setMessages((prev) => [...prev, assistantMsg]);
    }

    if (!overridePrompt) setQuestion("");
    setLoading(true);

    const historyForBackend = currentMessages.map((m) => ({
      role: m.role,
      content: m.content
    }));

    try {
      const response = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: promptToSend.trim(),
          document_id: documentId,
          allow_general: allowGeneralOverride ?? false,
          permission_response: permissionResponseOverride || null,
          messages: historyForBackend,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error("Cognivault chat request failed.");
      }

      setBackendStatus("online");
      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("application/x-ndjson") && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        let accumulatedAnswer = "";
        let buffer = "";

        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;
              try {
                const parsed = JSON.parse(line);
                if (parsed.type === "meta") {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                          ...msg,
                          source: parsed.source,
                          requires_permission: parsed.requires_permission,
                          conversation_state: parsed.conversation_state
                        }
                        : msg
                    )
                  );
                } else if (parsed.type === "token" && parsed.token) {
                  accumulatedAnswer += parsed.token;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: accumulatedAnswer }
                        : msg
                    )
                  );
                } else if (parsed.type === "metrics") {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, metrics: parsed.metrics }
                        : msg
                    )
                  );
                } else if (parsed.type === "error") {
                  console.error("Stream error:", parsed.error);
                }
              } catch (e) {
                console.error("JSON parse error on stream line:", e);
              }
            }
          }
        }
        // Flush any remaining buffer content after stream ends
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer.trim());
            if (parsed.type === "token" && parsed.token) {
              accumulatedAnswer += parsed.token;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: accumulatedAnswer }
                    : msg
                )
              );
            }
          } catch (e) {
            // Ignore parse errors on final buffer remnant
          }
        }
      } else {
        const data = await response.json();
        const finalAnswer = data.answer || "No answer was returned by Cognivault.";
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                ...msg,
                content: finalAnswer,
                source: data.source,
                requires_permission: data.requires_permission,
                conversation_state: data.conversation_state,
                metrics: data.metrics
              }
              : msg
          )
        );
      }
    } catch (error: any) {
      console.error("COGNIVAULT CHAT ERROR:", error);
      setBackendStatus("offline");
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              content: "⚠️ Error: " + (error.message || "Unable to connect to Cognivault backend. Please ensure the server is running at http://127.0.0.1:8000.")
            }
            : msg
        )
      );
    } finally {
      setLoading(false);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId ? { ...msg, isStreaming: false } : msg
        )
      );
    }
  };

  const handlePermissionChoice = (allow: boolean) => {
    if (allow) {
      askCognivault(undefined, true, "yes", true);
    } else {
      askCognivault(undefined, false, "no", true);
    }
  };

  // Keyboard shortcut (Enter to send, Shift+Enter for new line)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askCognivault();
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080b12] text-slate-100 antialiased font-sans">

      {/* =========================================================================
          LEFT SIDEBAR
      ========================================================================= */}
      <aside
        className={`relative flex flex-col border-r border-slate-800/80 bg-[#070a12] transition-all duration-300 ease-in-out z-20 ${isSidebarOpen ? "w-80 min-w-[20rem]" : "w-16 min-w-[4rem]"
          }`}
      >
        {/* Sidebar Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-800/60 px-4">
          {isSidebarOpen ? (
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 shadow-lg shadow-indigo-500/20">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                  Cognivault
                  <span className="rounded-full bg-indigo-500/10 px-2 py-0.5 text-[10px] font-semibold text-indigo-400 border border-indigo-500/20">
                    RAG v2.0
                  </span>
                </h1>
                <p className="text-[11px] text-slate-400 font-medium">Document Intelligence</p>
              </div>
            </div>
          ) : (
            <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
          )}

          {/* Sidebar Collapse Toggle */}
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 transition"
            title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isSidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </button>
        </div>

        {/* Sidebar Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* New Chat Button */}
          <button
            onClick={startNewChat}
            className={`flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-indigo-600/20 hover:bg-indigo-500 active:scale-[0.98] transition ${!isSidebarOpen && "px-2"
              }`}
          >
            <PlusCircle className="h-4 w-4" />
            {isSidebarOpen && <span>New Chat</span>}
          </button>

          {/* Document Management Section */}
          {isSidebarOpen && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
                <span>Active Knowledge Base</span>
                <FileText className="h-3.5 w-3.5 text-slate-500" />
              </div>

              {documentId ? (
                /* Active Indexed Document Card */
                <div className="group relative rounded-xl border border-indigo-500/30 bg-indigo-950/20 p-3.5 backdrop-blur-md transition hover:border-indigo-500/50">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2.5 overflow-hidden">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400">
                        <FileCheck className="h-4 w-4" />
                      </div>
                      <div className="overflow-hidden">
                        <p className="truncate text-xs font-semibold text-slate-200" title={uploadedFileName || "Document"}>
                          {uploadedFileName || "Uploaded PDF"}
                        </p>
                        <p className="text-[11px] text-indigo-300/70">{uploadedFileSize || "Indexed"}</p>
                      </div>
                    </div>

                    <button
                      onClick={clearDocument}
                      className="text-slate-500 hover:text-rose-400 transition p-1"
                      title="Clear Document"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  <div className="mt-2.5 flex items-center gap-1.5 text-[11px] font-medium text-emerald-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>Vector DB Ready (ChromaDB)</span>
                  </div>
                </div>
              ) : (
                /* Drag & Drop Upload Zone */
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  className={`group relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-4 text-center transition ${dragActive
                    ? "border-indigo-500 bg-indigo-500/10"
                    : "border-slate-800 hover:border-slate-700 bg-slate-900/40"
                    }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-slate-800 text-indigo-400 group-hover:scale-105 transition">
                    <UploadCloud className="h-5 w-5" />
                  </div>

                  {selectedFile ? (
                    <div className="space-y-2 w-full">
                      <p className="truncate text-xs font-medium text-slate-200" title={selectedFile.name}>
                        {selectedFile.name}
                      </p>
                      <button
                        onClick={uploadPDF}
                        disabled={uploading}
                        className="w-full rounded-lg bg-indigo-600 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 transition flex items-center justify-center gap-1.5"
                      >
                        {uploading ? (
                          <div className="flex flex-col items-center justify-center w-full">
                            <div className="flex items-center gap-1.5">
                                <RefreshCw className="h-3 w-3 animate-spin" />
                                <span className="truncate max-w-[150px]">{uploadStage || "Indexing..."}</span>
                            </div>
                            {uploadProgress > 0 && (
                                <div className="w-full bg-slate-700/50 rounded-full h-1 mt-1.5 overflow-hidden">
                                  <div className="bg-indigo-300 h-1 rounded-full transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
                                </div>
                            )}
                          </div>
                        ) : (
                          <span>Index PDF into Chroma</span>
                        )}
                      </button>
                    </div>
                  ) : (
                    <>
                      <p className="text-xs font-medium text-slate-300">
                        Drag PDF here or{" "}
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="text-indigo-400 hover:underline font-semibold"
                        >
                          browse
                        </button>
                      </p>
                      <p className="mt-1 text-[10px] text-slate-500">Supports PDF documents up to 50MB</p>
                    </>
                  )}
                </div>
              )}

              {/* Upload Status Message */}
              {uploadMessage && (
                <div
                  className={`rounded-lg p-2.5 text-xs font-medium ${uploadMessage.includes("successfully") || uploadMessage.includes("indexed")
                    ? "bg-emerald-950/40 text-emerald-300 border border-emerald-800/40"
                    : "bg-rose-950/40 text-rose-300 border border-rose-800/40"
                    }`}
                >
                  {uploadMessage}
                </div>
              )}
            </div>
          )}

          {/* Quick Info & Features */}
          {isSidebarOpen && (
            <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-3.5 space-y-2.5 text-xs text-slate-400">
              <div className="flex items-center gap-2 font-semibold text-slate-300">
                <ShieldCheck className="h-4 w-4 text-indigo-400" />
                <span>Grounded RAG Pipeline</span>
              </div>
              <ul className="space-y-1.5 text-[11px] list-disc list-inside text-slate-400">
                <li>Strict factual adherence</li>
                <li>MiniLM-L6 Vector Embeddings</li>
                <li>NDJSON Token Streaming</li>
              </ul>
            </div>
          )}
        </div>

        {/* Sidebar Footer: System Status */}
        <div className="border-t border-slate-800/60 p-3.5">
          {isSidebarOpen ? (
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Backend Server</span>
              <div className="flex items-center gap-1.5 font-medium">
                <span
                  className={`h-2 w-2 rounded-full ${backendStatus === "online"
                    ? "bg-emerald-400 shadow-sm shadow-emerald-400"
                    : backendStatus === "checking"
                      ? "bg-amber-400 animate-ping"
                      : "bg-rose-500"
                    }`}
                />
                <span
                  className={
                    backendStatus === "online"
                      ? "text-emerald-400"
                      : backendStatus === "checking"
                        ? "text-amber-400"
                        : "text-rose-400"
                  }
                >
                  {backendStatus === "online" ? "FastAPI Online" : backendStatus === "checking" ? "Checking" : "Offline"}
                </span>
              </div>
            </div>
          ) : (
            <div
              className={`mx-auto h-2.5 w-2.5 rounded-full ${backendStatus === "online" ? "bg-emerald-400" : "bg-rose-500"
                }`}
              title={`Backend ${backendStatus}`}
            />
          )}
        </div>
      </aside>

      {/* =========================================================================
          MAIN CONTENT / CHAT WORKSPACE
      ========================================================================= */}
      <main className="flex flex-1 flex-col overflow-hidden bg-[#080b12]">
        {/* Top Header Bar */}
        <header className="flex h-16 items-center justify-between border-b border-slate-800/80 px-6 glass-panel">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-200">
              {uploadedFileName ? (
                <span className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-indigo-400" />
                  <span className="truncate max-w-xs">{uploadedFileName}</span>
                </span>
              ) : (
                "Academic Document Assistant"
              )}
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/analytics" className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition">
              <BarChart3 className="h-3.5 w-3.5" /> Analytics
            </Link>
            <Link href="/architecture" className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition">
              <Network className="h-3.5 w-3.5" /> Architecture
            </Link>
            <div className="w-px h-4 bg-slate-800/80 hidden md:block mx-1"></div>

            {documentId ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400 border border-emerald-500/20">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Document Connected
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-400 border border-amber-500/20">
                <AlertCircle className="h-3.5 w-3.5" />
                Upload PDF to Query
              </span>
            )}
          </div>
        </header>

        {/* Chat Thread Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 lg:px-12 space-y-6">
          {messages.length === 0 ? (
            /* Welcome / Empty Conversation State */
            <div className="flex h-full flex-col items-center justify-center text-center max-w-2xl mx-auto py-12">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 mb-6 glow-indigo">
                <Sparkles className="h-8 w-8" />
              </div>

              <h2 className="text-2xl font-bold tracking-tight text-white mb-2">
                Ask Cognivault Anything
              </h2>
              <p className="text-sm text-slate-400 mb-8 leading-relaxed">
                Upload English academic papers, lecture notes, or textbooks to ask questions with exact semantic vector retrieval. (Note: Cognivault currently supports English only)
              </p>

              {/* Sample Prompt Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 w-full">
                {[
                  {
                    title: "Summarize Main Points",
                    desc: "Provide a structured breakdown of key concepts.",
                    prompt: "What is this PDF about?"
                  },
                  {
                    title: "Extract Key Topics",
                    desc: "List core subjects covered in the text.",
                    prompt: "What topics are covered in this document?"
                  },
                  {
                    title: "Multiple Choice Questions",
                    desc: "Generate practice MCQs with correct options.",
                    prompt: "Generate practice multiple choice questions from this document."
                  },
                  {
                    title: "Explain Complex Concepts",
                    desc: "Detail the mechanisms and explanations.",
                    prompt: "Explain the main mechanisms discussed in the document in detail."
                  }
                ].map((card, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setQuestion(card.prompt);
                      askCognivault(card.prompt);
                    }}
                    className="flex flex-col items-start p-4 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:bg-slate-800/60 hover:border-indigo-500/30 text-left transition group"
                  >
                    <div className="flex items-center justify-between w-full mb-1">
                      <span className="text-xs font-semibold text-slate-200 group-hover:text-indigo-400 transition">
                        {card.title}
                      </span>
                      <Zap className="h-3.5 w-3.5 text-slate-500 group-hover:text-indigo-400 transition" />
                    </div>
                    <span className="text-[11px] text-slate-400">{card.desc}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Messages List */
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-4 max-w-4xl mx-auto ${msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
              >
                {msg.role === "assistant" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 text-white shadow-md shadow-indigo-600/20 mt-1">
                    <Bot className="h-4.5 w-4.5" />
                  </div>
                )}

                <div
                  className={`relative flex flex-col group max-w-[85%] rounded-2xl p-4 transition-all ${msg.role === "user"
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/10 rounded-tr-xs"
                    : "glass-panel text-slate-200 border border-slate-800/80 rounded-tl-xs"
                    }`}
                >
                  {/* Role & Actions Bar */}
                  <div className="flex items-center justify-between mb-2 pb-1 border-b border-white/5 text-[11px] text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold tracking-wide text-slate-200">
                        {msg.role === "user" ? "You" : "Cognivault AI"}
                      </span>

                      {/* Source Indicator Badge */}
                      {msg.role === "assistant" && msg.source && (
                        <div className="flex items-center gap-1.5 text-[10px] font-medium ml-1">
                          {msg.source === "general_knowledge" || msg.source === "general" ? (
                            <span className="inline-flex items-center gap-1 text-amber-300 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
                              🌐 General knowledge
                            </span>
                          ) : msg.source === "conversation" ? (
                            <span className="inline-flex items-center gap-1 text-blue-300 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded-full">
                              💬 Conversation
                            </span>
                          ) : msg.source === "document_not_found" ? (
                            <span className="inline-flex items-center gap-1 text-slate-300 bg-slate-500/10 border border-slate-500/20 px-2 py-0.5 rounded-full">
                              📄 Document
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                              📄 Document-grounded
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[10px] opacity-70">{msg.timestamp}</span>

                      {msg.role === "assistant" && (
                        <button
                          onClick={() => copyToClipboard(msg.id, msg.content)}
                          className="text-slate-400 hover:text-white transition p-0.5 rounded"
                          title="Copy response"
                        >
                          {copiedMessageId === msg.id ? (
                            <Check className="h-3.5 w-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="h-3.5 w-3.5" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Message Content */}
                  {msg.role === "assistant" ? (
                    <div className={`prose-dark text-sm ${msg.isStreaming ? "streaming-cursor" : ""}`}>
                      <ReactMarkdown 
                        components={{ 
                          p: ({node, ...props}) => <p className="whitespace-pre-wrap mb-4" {...props} /> 
                        }}
                      >
                        {(msg.content || "...").replace(/\n\s*([A-D]\))/g, '\n\n$1')}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm font-medium whitespace-pre-wrap leading-relaxed">
                      {msg.content}
                    </p>
                  )}

                  {/* Interactive Action Buttons for Permission Request */}
                  {msg.role === "assistant" && msg.requires_permission && !msg.isStreaming && (
                    <div className="mt-3 pt-3 border-t border-slate-700/60 flex flex-wrap items-center gap-2.5">
                      <button
                        onClick={() => handlePermissionChoice(true)}
                        className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-3.5 py-2 text-xs font-semibold text-white shadow-md shadow-indigo-600/20 hover:from-indigo-500 hover:to-violet-500 active:scale-[0.98] transition"
                      >
                        <span>🌐 Yes, give me a general answer</span>
                      </button>
                      <button
                        onClick={() => handlePermissionChoice(false)}
                        className="flex items-center gap-1.5 rounded-xl bg-slate-800/90 border border-slate-700/80 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white active:scale-[0.98] transition"
                      >
                        <span>📄 No, keep it document-grounded</span>
                      </button>
                    </div>
                  )}

                  {/* Profiling Metrics (if present) */}
                  {msg.metrics && msg.metrics.total_api_time && (
                    <div className="mt-3 pt-2 border-t border-white/5 flex flex-wrap items-center gap-3 text-[10px] text-slate-400 font-mono">
                      <span>Retrieval: {msg.metrics.chromadb_retrieval_time}s</span>
                      <span>TTFT: {msg.metrics.first_token_time}s</span>
                      <span>Generation: {msg.metrics.llm_generation_time}s</span>
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-slate-800 text-slate-300 mt-1">
                    <User className="h-4.5 w-4.5" />
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input Footer Bar */}
        <div className="p-4 md:px-8 lg:px-12 border-t border-slate-800/80 glass-panel">
          <div className="max-w-4xl mx-auto space-y-2">
            {/* Input Box Container */}
            <div className="relative flex items-end rounded-2xl border border-slate-800 bg-slate-900/90 p-2 shadow-2xl focus-within:border-indigo-500/60 focus-within:ring-2 focus-within:ring-indigo-500/20 transition">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  documentId
                    ? "Ask a question about your uploaded document..."
                    : "Ask Cognivault or upload a PDF to analyze..."
                }
                rows={1}
                className="flex-1 max-h-32 min-h-[44px] resize-none bg-transparent px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
              />

              <button
                onClick={() => askCognivault()}
                disabled={loading || !question.trim()}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-md shadow-indigo-600/20 hover:bg-indigo-500 active:scale-95 disabled:opacity-40 disabled:hover:bg-indigo-600 transition ml-2"
                title="Send Message"
              >
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </button>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-400 px-2">
              <span className="flex items-center gap-1">
                <HelpCircle className="h-3 w-3 text-slate-400" />
                Press <kbd className="rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300 font-mono">Enter</kbd> to send, <kbd className="rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300 font-mono">Shift+Enter</kbd> for line break
              </span>
              <span>Cognivault RAG Pipeline v2.0 (English Only)</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
