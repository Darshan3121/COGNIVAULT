"use client";

import { useState, useEffect } from "react";
import {
  Sparkles,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  MessageSquare,
  Network,
  AlertCircle
} from "lucide-react";
import Link from "next/link";

export default function Analytics() {
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedId = sessionStorage.getItem("cognivault_documentId");
      if (storedId) {
        setDocumentId(storedId);
        setUploadedFileName(sessionStorage.getItem("cognivault_fileName") || null);
        fetchMetrics(storedId);
      }
    }
  }, []);

  const fetchMetrics = async (docId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/analytics/${docId}`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080b12] text-slate-100 antialiased font-sans">
      <aside className={`relative flex flex-col border-r border-slate-800/80 bg-[#070a12] transition-all duration-300 ease-in-out z-20 ${isSidebarOpen ? "w-80 min-w-[20rem]" : "w-16 min-w-[4rem]"}`}>
        {/* Sidebar Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-800/60 px-4">
          {isSidebarOpen ? (
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 shadow-lg shadow-emerald-500/20">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">Cognivault</h1>
                <p className="text-[11px] text-slate-400 font-medium">Analytics Engine</p>
              </div>
            </div>
          ) : (
            <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500">
              <BarChart3 className="h-5 w-5 text-white" />
            </div>
          )}
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 transition">
            {isSidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </button>
        </div>

        {/* Sidebar Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {isSidebarOpen && (
            <div className="pt-2 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
                <span>Navigation</span>
              </div>
              <Link href="/" className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-300 hover:bg-slate-800/60 hover:text-slate-100 transition font-medium">
                <MessageSquare className="h-4 w-4 text-indigo-400" />
                <span>Chat</span>
              </Link>
              <Link href="/analytics" className="flex items-center gap-3 px-3 py-2 rounded-lg bg-emerald-600/20 text-emerald-400 font-medium border border-emerald-500/30">
                <BarChart3 className="h-4 w-4" />
                <span>Analytics</span>
              </Link>
              <Link href="/architecture" className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-300 hover:bg-slate-800/60 hover:text-slate-100 transition font-medium">
                <Network className="h-4 w-4 text-amber-400" />
                <span>Architecture</span>
              </Link>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex flex-1 flex-col overflow-hidden bg-gradient-to-br from-[#080b12] to-[#0f172a]">
        <header className="flex h-16 items-center justify-between border-b border-slate-800/80 px-6 glass-panel">
          <h2 className="text-sm font-semibold text-slate-200">
            {uploadedFileName ? `Analytics for ${uploadedFileName}` : "Document Analytics"}
          </h2>
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-5xl mx-auto">
            <h1 className="text-3xl font-bold text-white mb-8 tracking-tight">Analytics Dashboard</h1>
            
            {!documentId ? (
              <div className="flex flex-col items-center justify-center p-12 text-center border border-slate-800 bg-slate-900/50 rounded-2xl">
                <AlertCircle className="h-12 w-12 text-amber-500 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">No Document Loaded</h3>
                <p className="text-slate-400 mb-6">Please upload a document on the Chat page to view its analytics.</p>
                <Link href="/" className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-500 transition shadow-lg shadow-indigo-600/20">
                  Go to Chat
                </Link>
              </div>
            ) : loading ? (
              <div className="flex justify-center items-center h-64">
                <div className="animate-spin h-8 w-8 border-4 border-emerald-500 border-t-transparent rounded-full"></div>
              </div>
            ) : metrics ? (
              <div className="space-y-8">
                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-slate-900/80 border border-emerald-500/30 rounded-2xl p-6 shadow-lg shadow-emerald-900/20 backdrop-blur-sm hover:border-emerald-500/60 transition-colors">
                    <p className="text-sm font-semibold text-slate-400 mb-1">Document Health</p>
                    <p className="text-4xl font-extrabold text-emerald-400">{metrics.word_count?.toLocaleString() || 0}</p>
                    <p className="text-xs text-slate-500 mt-2 uppercase tracking-wide">Words Indexed</p>
                  </div>
                  
                  <div className="bg-slate-900/80 border border-indigo-500/30 rounded-2xl p-6 shadow-lg shadow-indigo-900/20 backdrop-blur-sm hover:border-indigo-500/60 transition-colors">
                    <p className="text-sm font-semibold text-slate-400 mb-1">AI Processing</p>
                    <p className="text-4xl font-extrabold text-indigo-400">{metrics.num_chunks?.toLocaleString() || 0}</p>
                    <p className="text-xs text-slate-500 mt-2 uppercase tracking-wide">ChromaDB Chunks</p>
                  </div>
                  
                  <div className="bg-slate-900/80 border border-amber-500/30 rounded-2xl p-6 shadow-lg shadow-amber-900/20 backdrop-blur-sm hover:border-amber-500/60 transition-colors">
                    <p className="text-sm font-semibold text-slate-400 mb-1">Total Characters</p>
                    <p className="text-4xl font-extrabold text-amber-400">{metrics.char_count?.toLocaleString() || 0}</p>
                    <p className="text-xs text-slate-500 mt-2 uppercase tracking-wide">Raw Content</p>
                  </div>
                </div>

                {/* Insights Section */}
                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm">
                  <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <Sparkles className="h-6 w-6 text-amber-400" />
                    Key Document Insights
                  </h3>
                  
                  {metrics.top_insights && metrics.top_insights.length > 0 ? (
                    <div className="flex flex-wrap gap-3">
                      {metrics.top_insights.map((insight: string, idx: number) => (
                        <span key={idx} className="px-4 py-2 bg-indigo-900/40 border border-indigo-500/50 text-indigo-300 rounded-full font-medium text-sm shadow-sm hover:bg-indigo-800/60 transition cursor-default">
                          {insight}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-400">No primary business or technical themes detected.</p>
                  )}
                  
                  <div className="mt-8 p-5 bg-emerald-900/20 border-l-4 border-emerald-500 rounded-r-lg">
                    <p className="text-emerald-300 text-sm font-medium leading-relaxed">
                      <strong>🤖 AI Logic:</strong> This document emphasizes {metrics.top_insights && metrics.top_insights.length > 0 ? metrics.top_insights.slice(0, 2).join(" and ") : "general concepts"} based on chunk density analysis.
                    </p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </main>
    </div>
  );
}
