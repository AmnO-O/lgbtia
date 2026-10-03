import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import { 
  Terminal, 
  Cpu, 
  Layers, 
  CheckCircle2, 
  Copy, 
  Play, 
  ShieldAlert, 
  Sparkles,
  Database,
  Code2
} from 'lucide-react';

function App() {
  const [copied, setCopied] = useState<string | null>(null);
  const [modelName, setModelName] = useState('microsoft/mdeberta-v3-base');
  const [batchSize, setBatchSize] = useState(16);
  const [epochs, setEpochs] = useState(8);
  const [lr, setLr] = useState(0.00002);

  const trainCmd = `python run_pipeline.py --mode train --model_name "${modelName}" --batch_size ${batchSize} --epochs ${epochs} --lr ${lr}`;
  const predictCmd = `python run_pipeline.py --mode predict --checkpoint "./outputs_task_b/best_task_b_model.pt" --test_tsv "StereoQueerEval_TEST.tsv"`;

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600/20 text-indigo-400 rounded-lg border border-indigo-500/30">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                StereoQueerEval <span className="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-500/30 font-medium">Task B: Hate Speech</span>
              </h1>
              <p className="text-xs text-slate-400">Multilingual 3-Class Classification (EN / IT / NL / FA)</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="flex items-center gap-1.5 px-3 py-1 bg-emerald-950/60 text-emerald-400 rounded-md border border-emerald-800/40">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Backend Ready
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Classes Banner */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Class 0</span>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">0</span>
            </div>
            <h3 className="text-lg font-bold text-slate-100 mb-1">no</h3>
            <p className="text-xs text-slate-400">Standard comments without hateful or discriminatory intent.</p>
          </div>

          <div className="bg-slate-900/80 border border-amber-900/40 rounded-xl p-5 hover:border-amber-700/60 transition">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-amber-400">Class 1</span>
              <span className="text-xs px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 font-mono">1</span>
            </div>
            <h3 className="text-lg font-bold text-amber-200 mb-1">yes_implicit</h3>
            <p className="text-xs text-slate-400">Subtle, veiled, coded, or context-dependent hateful content.</p>
          </div>

          <div className="bg-slate-900/80 border border-rose-900/40 rounded-xl p-5 hover:border-rose-700/60 transition">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-rose-400">Class 2</span>
              <span className="text-xs px-2 py-0.5 rounded bg-rose-950/80 text-rose-300 font-mono">2</span>
            </div>
            <h3 className="text-lg font-bold text-rose-200 mb-1">yes_explicit</h3>
            <p className="text-xs text-slate-400">Direct slurs, dehumanization, harassment, or incitement.</p>
          </div>
        </div>

        {/* Training Parameters & CLI Generator */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-4">
            <Terminal className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-slate-100">CLI Training & Inference Commands</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Model Backbone</label>
              <select 
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-xs"
              >
                <option value="microsoft/mdeberta-v3-base">microsoft/mdeberta-v3-base</option>
                <option value="xlm-roberta-base">xlm-roberta-base</option>
                <option value="jhu-clsp/mmbert-base">jhu-clsp/mmbert-base (ModernBERT)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Batch Size</label>
              <input 
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-xs"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Epochs</label>
              <input 
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-xs"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Learning Rate</label>
              <input 
                type="number"
                step="0.00001"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-xs"
              />
            </div>
          </div>

          {/* Generated Command 1 */}
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-medium">1. Run Training:</span>
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex items-center justify-between font-mono text-xs text-slate-300">
              <span className="overflow-x-auto whitespace-nowrap mr-4">{trainCmd}</span>
              <button 
                onClick={() => copyToClipboard(trainCmd, 'train')}
                className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition flex items-center gap-1 shrink-0"
              >
                {copied === 'train' ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Generated Command 2 */}
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-medium">2. Run Inference / Test Prediction:</span>
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex items-center justify-between font-mono text-xs text-slate-300">
              <span className="overflow-x-auto whitespace-nowrap mr-4">{predictCmd}</span>
              <button 
                onClick={() => copyToClipboard(predictCmd, 'predict')}
                className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition flex items-center gap-1 shrink-0"
              >
                {copied === 'predict' ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>

        {/* Architecture Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-indigo-400 text-sm font-semibold">
              <Database className="w-4 h-4" />
              <span>Input Representation & Context Triplet</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Comments often depend on the context of the video. The dataset concatenates:
            </p>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-xs text-indigo-300">
              Comment: &lt;yt_comment&gt; [SEP] Video: &lt;yt_title&gt; [SEP] Description: &lt;yt_description&gt;
            </div>
            <p className="text-xs text-slate-400">
              Multi-lingual diacritics and accented characters (Italian: <code className="text-slate-300 font-mono">perché</code>, <code className="text-slate-300 font-mono">è</code>; Dutch; Persian) are preserved in full.
            </p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-semibold">
              <ShieldAlert className="w-4 h-4" />
              <span>Leakage-Free Validation & Imbalance Control</span>
            </div>
            <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside">
              <li>
                <strong className="text-slate-100">Grouped Split:</strong> Grouped by <code className="text-slate-300 font-mono">yt_title</code> to ensure no comments from the same video are present in both train and validation sets.
              </li>
              <li>
                <strong className="text-slate-100">Class Weighting:</strong> Automatically computes balanced class weights to compensate for severe class imbalance.
              </li>
              <li>
                <strong className="text-slate-100">Macro-F1 Tracking:</strong> Early stopping tracks validation Macro-F1 across the 3 classes.
              </li>
            </ul>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 text-center text-xs text-slate-400">
        StereoQueerEval — Task B NLP Codebase &bull; Ready for Colab, Kaggle, or Local GPU training.
      </footer>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
