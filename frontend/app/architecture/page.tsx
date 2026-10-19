"use client";

import { useState } from "react";
import {
  Sparkles,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  MessageSquare,
  Network,
  Cpu,
  Database,
  SearchCode
} from "lucide-react";
import Link from "next/link";

export default function Architecture() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080b12] text-slate-100 antialiased font-sans">
      <aside className={`relative flex flex-col border-r border-slate-800/80 bg-[#070a12] transition-all duration-300 ease-in-out z-20 ${isSidebarOpen ? "w-80 min-w-[20rem]" : "w-16 min-w-[4rem]"}`}>
        {/* Sidebar Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-800/60 px-4">
          {isSidebarOpen ? (
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-600 to-orange-500 shadow-lg shadow-amber-500/20">
                <Network className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">Cognivault</h1>
                <p className="text-[11px] text-slate-400 font-medium">Architecture Team</p>
              </div>
            </div>
          ) : (
            <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-600 to-orange-500">
              <Network className="h-5 w-5 text-white" />
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
              <Link href="/analytics" className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-300 hover:bg-slate-800/60 hover:text-slate-100 transition font-medium">
                <BarChart3 className="h-4 w-4 text-emerald-400" />
                <span>Analytics</span>
              </Link>
              <Link href="/architecture" className="flex items-center gap-3 px-3 py-2 rounded-lg bg-amber-600/20 text-amber-400 font-medium border border-amber-500/30">
                <Network className="h-4 w-4" />
                <span>Architecture</span>
              </Link>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex flex-1 flex-col overflow-y-auto bg-[#080b12] relative">
        {/* Animated Background Gradients */}
        <div className="absolute top-0 left-0 w-full h-96 bg-indigo-900/10 rounded-full blur-[100px] pointer-events-none"></div>
        <div className="absolute bottom-0 right-0 w-2/3 h-96 bg-amber-900/10 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="px-8 py-12 md:py-20 max-w-5xl mx-auto w-full relative z-10">
          <div className="bg-indigo-600/10 border-l-4 border-indigo-500 p-8 rounded-2xl mb-16 backdrop-blur-md border-y border-r border-indigo-500/20 shadow-2xl">
            <h2 className="text-3xl font-extrabold text-white mb-3">Mission</h2>
            <p className="text-xl text-indigo-200 font-medium">
              Solving information overload by transforming static documents into queryable intelligence.
            </p>
          </div>

          <h3 className="text-2xl font-bold text-white mb-8">The Engine</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            <div className="bg-slate-900/80 border border-slate-700 hover:border-indigo-500 transition duration-300 rounded-2xl p-8 group shadow-lg shadow-black/50">
              <Cpu className="h-12 w-12 text-indigo-400 mb-6 group-hover:scale-110 transition duration-300" />
              <h4 className="text-xl font-bold text-white mb-3">Brain: Mistral 7B</h4>
              <p className="text-slate-400 leading-relaxed">
                Powered by advanced LLMs via Ollama for sub-second generative inference and complex reasoning.
              </p>
            </div>
            
            <div className="bg-slate-900/80 border border-slate-700 hover:border-emerald-500 transition duration-300 rounded-2xl p-8 group shadow-lg shadow-black/50">
              <Database className="h-12 w-12 text-emerald-400 mb-6 group-hover:scale-110 transition duration-300" />
              <h4 className="text-xl font-bold text-white mb-3">Memory: ChromaDB</h4>
              <p className="text-slate-400 leading-relaxed">
                Vector Store optimized for high-speed, scalable similarity search across document chunks.
              </p>
            </div>
            
            <div className="bg-slate-900/80 border border-slate-700 hover:border-amber-500 transition duration-300 rounded-2xl p-8 group shadow-lg shadow-black/50">
              <SearchCode className="h-12 w-12 text-amber-400 mb-6 group-hover:scale-110 transition duration-300" />
              <h4 className="text-xl font-bold text-white mb-3">Translator: MiniLM-L6</h4>
              <p className="text-slate-400 leading-relaxed">
                Industry-standard HuggingFace model for generating dense semantic vector embeddings.
              </p>
            </div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-10 mb-16 shadow-xl">
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Sparkles className="h-6 w-6 text-amber-400" />
              Advanced Capabilities
            </h3>
            <ul className="space-y-6">
              <li className="flex items-start gap-4">
                <div className="bg-indigo-500/20 p-2 rounded-lg mt-1 border border-indigo-500/30">
                  <div className="h-2 w-2 bg-indigo-400 rounded-full"></div>
                </div>
                <div>
                  <strong className="text-indigo-300 text-lg block mb-1">Intelligent Chunking:</strong>
                  <span className="text-slate-400">Ensures context is perfectly preserved and relationships maintained across page breaks and multiple choice questions.</span>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <div className="bg-emerald-500/20 p-2 rounded-lg mt-1 border border-emerald-500/30">
                  <div className="h-2 w-2 bg-emerald-400 rounded-full"></div>
                </div>
                <div>
                  <strong className="text-emerald-300 text-lg block mb-1">Context-Aware Retrieval:</strong>
                  <span className="text-slate-400">Only sends the absolute most relevant document chunks to the LLM to aggressively prevent hallucinations and ground responses.</span>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <div className="bg-amber-500/20 p-2 rounded-lg mt-1 border border-amber-500/30">
                  <div className="h-2 w-2 bg-amber-400 rounded-full"></div>
                </div>
                <div>
                  <strong className="text-amber-300 text-lg block mb-1">NDJSON Streaming:</strong>
                  <span className="text-slate-400">Leveraging high-performance FastAPI backends and streaming tokens to achieve lightning-fast user experiences.</span>
                </div>
              </li>
            </ul>
          </div>

          <div className="flex justify-center mb-12">
            <div className="bg-emerald-900/30 border border-emerald-500/50 text-emerald-400 px-6 py-3 rounded-full font-bold flex items-center gap-3 shadow-[0_0_20px_rgba(16,185,129,0.2)]">
              <div className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </div>
              SYSTEM STATUS: PRODUCTION READY | ULTRA-LOW LATENCY
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
