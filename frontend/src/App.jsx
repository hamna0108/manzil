import React, { useState, useEffect, useCallback, useRef } from "react";

// ============================================================================
// ICONS (Zero Dependencies)
// ============================================================================
const ManzilMark = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" fill="none" className="text-amber-500 dark:text-amber-400">
    <path d="M6 30 L24 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    <path d="M42 30 L26.5 12.5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    <path d="M6 30 L6 38" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    <path d="M42 30 L42 38" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    <circle cx="24" cy="20.5" r="4" fill="currentColor" />
  </svg>
);
const CheckCircle = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-500"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>;
const SearchIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>;
const BookmarkIcon = ({ filled }) => <svg width="20" height="20" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={filled ? "text-amber-500 dark:text-amber-400" : "text-slate-400 dark:text-slate-500 hover:text-amber-500 dark:hover:text-amber-400"}><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/></svg>;
const MapPinIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>;
const XIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>;
const SparkleIcon = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1-1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>;
const TrendingUp = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>;
const ExternalLinkIcon = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>;

// --- NEW MIC ICON ---
const MicIcon = ({ isRecording }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill={isRecording ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    <line x1="12" x2="12" y1="19" y2="22"/>
  </svg>
);

const API_BASE_URL = "http://127.0.0.1:8000";

export default function App() {
  const [phase, setPhase] = useState("hero"); 
  const [query, setQuery] = useState("");
  const [clarification, setClarification] = useState(null); 
  const [clarificationIntent, setClarificationIntent] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState("");
  const [activeProperty, setActiveProperty] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  // --- AUDIO RECORDING STATE ---
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);

  // --- AUTH STATE ---
  const [token, setToken] = useState(() => localStorage.getItem("userToken") || null);
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login"); 
  const [authForm, setAuthForm] = useState({ email: "", password: "" });
  const [authError, setAuthError] = useState("");
  
  // --- SAVED PROPERTIES STATE ---
  const [savedProps, setSavedProps] = useState([]);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const formatPrice = (price) => {
    if (!price) return "Price on Call";
    if (price >= 10000000) return `${(price / 10000000).toFixed(2)} Crore`;
    if (price >= 100000) return `${(price / 100000).toFixed(2)} Lakh`;
    return price.toLocaleString();
  };

  const fetchSavedProperties = useCallback(async (currentToken) => {
    if (!currentToken) return;
    try {
      const response = await fetch(`${API_BASE_URL}/saved-listings`, {
        headers: { "Authorization": `Bearer ${currentToken}` }
      });
      
      if (response.status === 401) {
        localStorage.removeItem("userToken");
        setToken(null);
        setSavedProps([]);
        return;
      }

      if (response.ok) {
        const data = await response.json();
        setSavedProps(data || []);
      }
    } catch (error) {
      console.error("Failed to fetch saved properties:", error);
    }
  }, []);

  useEffect(() => {
    if (token) {
      fetchSavedProperties(token);
    }
  }, [token, fetchSavedProperties]);

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError("");
    
    try {
      if (authMode === "signup") {
        const signupRes = await fetch(`${API_BASE_URL}/auth/signup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: authForm.email, password: authForm.password }),
        });
        
        if (!signupRes.ok) {
          const errData = await signupRes.json();
          const msg = typeof errData.detail === 'string' 
            ? errData.detail 
            : (Array.isArray(errData.detail) ? errData.detail[0].msg : "Signup failed.");
          throw new Error(msg);
        }
      }
      
      const loginRes = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          email: authForm.email, 
          password: authForm.password 
        }),
      });

      if (!loginRes.ok) {
        const errData = await loginRes.json();
        const msg = typeof errData.detail === 'string' 
          ? errData.detail 
          : (Array.isArray(errData.detail) ? errData.detail[0].msg : "Login failed.");
        throw new Error(msg);
      }

      const data = await loginRes.json();
      const userToken = data.access_token; 
      
      localStorage.setItem("userToken", userToken);
      setToken(userToken);
      setShowAuth(false);
      setAuthForm({ email: "", password: "" });
      
      fetchSavedProperties(userToken);

    } catch (error) {
      setAuthError(error.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("userToken");
    setToken(null);
    setSavedProps([]); 
    if (phase === "saved") setPhase("hero");
  };

const toggleSave = async (e, property) => {
    e.stopPropagation(); 
    if (!token) {
      setShowAuth(true);
      return;
    }
    
    // Hamesha real property ID nikalen, chahe wo search se aaye ya saved DB se
    const targetId = property.listing_id || property.id;
    
    const alreadySaved = savedProps.some(p => String(p.listing_id || p.id) === String(targetId));
    
    setSavedProps(prev => alreadySaved 
      ? prev.filter(p => String(p.listing_id || p.id) !== String(targetId)) 
      : [...prev, property]
    );

    try {
      let response;
      if (alreadySaved) {
        // DELETE request mein bhi targetId use karein
        response = await fetch(`${API_BASE_URL}/saved-listings/${targetId}`, {
          method: "DELETE",
          headers: { "Authorization": `Bearer ${token}` }
        });
      } else {
        response = await fetch(`${API_BASE_URL}/saved-listings`, {
          method: "POST",
          headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
          body: JSON.stringify({ listing_id: targetId, ...property })
        });
      }
      if (!response.ok) throw new Error("Sync failed");
    } catch (error) {
      fetchSavedProperties(token);
    }
  };

  const isSaved = (id) => savedProps.some(p => String(p.listing_id || p.id) === String(id));

  // ============================================================================
  // AUDIO RECORDING LOGIC
  // ============================================================================
  const sendAudioToBackend = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_query.ogg');

    try {
      setErrorMessage("");
      const response = await fetch(`${API_BASE_URL}/voice/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Transcription failed");
      }

      const data = await response.json();

      if (data.transcript && !data.no_speech_detected) {
        setQuery((prevText) => 
          prevText ? `${prevText} ${data.transcript}` : data.transcript
        );
      }
    } catch (error) {
      console.error("Transcription failed:", error);
      setErrorMessage("Could not transcribe audio. Please try again.");
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop Recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
      }
    } else {
      // Start Recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Force supported format
        const options = { mimeType: 'audio/ogg;codecs=opus' };
        const recorder = new MediaRecorder(
          stream, 
          MediaRecorder.isTypeSupported(options.mimeType) ? options : undefined
        );
        
        mediaRecorderRef.current = recorder;
        const chunks = [];
        
        recorder.ondataavailable = (e) => chunks.push(e.data);
        
        recorder.onstop = async () => {
          const audioBlob = new Blob(chunks, { type: recorder.mimeType || 'audio/ogg' });
          await sendAudioToBackend(audioBlob);
          // Release the microphone
          stream.getTracks().forEach(track => track.stop());
        };

        recorder.start();
        setIsRecording(true);
      } catch (error) {
        console.error("Microphone access denied:", error);
        setErrorMessage("Microphone access denied. Please allow permissions in your browser.");
      }
    }
  };

  // ============================================================================
  // SEARCH EXECUTION
  // ============================================================================
  const executeSearch = async (searchString, intentOverride = null, clarificationField = null, clarificationValue = null) => {
    setPhase("thinking");
    setErrorMessage("");

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchString,
          intent_override: intentOverride,
          clarification_field: clarificationField,
          clarification_value: clarificationValue,
        }),
      });

      if (!response.ok) throw new Error("Backend connection failed.");

      const data = await response.json();
      
      if (data.clarification) {
        setClarification(data.clarification);
        setClarificationIntent(data.intent);
        setPhase("clarify");
        return;
      }
      
      const displaySummary = data.summary || data.gemini_summary || data.message || "Here are the best matches found by the Manzil Intelligence Engine.";
      
      setSummary(displaySummary);
      setResults(data.results || []);
      setClarification(null);
      setClarificationIntent(null);
      
      if (data.results && data.results.length > 0) {
        setActiveProperty(data.results[0]);
      }
      setPhase("results");

    } catch (error) {
      setErrorMessage("Could not connect to the Manzil AI engine. Ensure your FastAPI server is running.");
      setPhase("hero"); 
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) executeSearch(query);
  };

  const PropertyCard = ({ r }) => {
    const getPoiText = () => {
      if (r.poi_verification && Object.keys(r.poi_verification).length > 0) {
        const verifiedPois = [];
        const unverifiedPois = [];

        Object.entries(r.poi_verification).forEach(([type, data]) => {
          const cleanType = type.charAt(0).toUpperCase() + type.slice(1);

          if (data.verified === true) {
            if (data.name && data.name.toLowerCase() !== type.toLowerCase()) {
              verifiedPois.push(`${data.name} (${cleanType})`);
            } else {
              verifiedPois.push(cleanType);
            }
          } else if (data.verified === false) {
            unverifiedPois.push(cleanType);
          }
        });

        if (verifiedPois.length > 0) return { text: verifiedPois.join(", "), status: "found" };
        if (unverifiedPois.length > 0) return { text: `No ${unverifiedPois.join(", ")} found`, status: "missing" };
      }
      return null; 
    };

    const poiData = getPoiText();

    return (
      <div 
        onClick={() => setActiveProperty(r)}
        className={`group bg-white dark:bg-[#18181b] border-2 rounded-2xl p-5 transition-all cursor-pointer 
          ${activeProperty && activeProperty.id === r.id 
            ? 'border-amber-500 shadow-lg shadow-amber-100 dark:shadow-none' 
            : 'border-slate-100 dark:border-white/5 hover:border-amber-300 dark:hover:border-amber-500/50 hover:shadow-sm'}`}
      >
        <div className="flex justify-between items-start mb-4">
          <div className="flex flex-wrap gap-2">
            {r.property_type && (
              <span className="px-3 py-1 rounded-md bg-amber-100 dark:bg-amber-500/20 text-xs font-bold text-amber-700 dark:text-amber-400">
                {r.property_type}
              </span>
            )}
            <span className="px-3 py-1 rounded-md bg-slate-100 dark:bg-white/5 text-xs font-bold text-slate-700 dark:text-slate-300">{r.area_marla} Marla</span>
            {r.bedrooms > 0 && <span className="px-3 py-1 rounded-md bg-slate-100 dark:bg-white/5 text-xs font-bold text-slate-700 dark:text-slate-300">{r.bedrooms} Beds</span>}
            {r.bathrooms > 0 && <span className="px-3 py-1 rounded-md bg-slate-100 dark:bg-white/5 text-xs font-bold text-slate-700 dark:text-slate-300">{r.bathrooms} Baths</span>}
            
            {poiData && poiData.status === "missing" ? (
              <span className="px-3 py-1 rounded-md bg-red-50 dark:bg-red-500/10 text-xs font-bold text-red-700 dark:text-red-400 border border-red-200 dark:border-red-500/20 capitalize">
                ❌ {poiData.text}
              </span>
            ) : poiData ? (
              <span className="px-3 py-1 rounded-md bg-blue-50 dark:bg-blue-500/10 text-xs font-bold text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20 capitalize">
                📍 Near {poiData.text}
              </span>
            ) : null}
          </div>
          <button onClick={(e) => toggleSave(e, r)} className="p-1">
            <BookmarkIcon filled={isSaved(r.listing_id || r.id)} />
          </button>
        </div>
        
        <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1.5 leading-tight">{r.title}</h3>
        <p className="flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400 mb-5"><MapPinIcon /> {r.location}</p>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-4 border-t border-slate-100 dark:border-white/5">
          <span className="text-2xl font-black text-slate-900 dark:text-white">{formatPrice(r.price_pkr)}</span>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400 font-bold bg-emerald-50 dark:bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-200 dark:border-emerald-500/20">
              <CheckCircle /> Verified
            </span>
            {r.url && (
              <a href={r.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-xs font-bold text-slate-500 hover:text-amber-500 transition-colors" onClick={(e) => e.stopPropagation()}>
                Source <ExternalLinkIcon />
              </a>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen font-sans bg-slate-50 text-slate-900 dark:bg-[#09090b] dark:text-slate-100 transition-colors duration-300 selection:bg-amber-500/30 relative">
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-no-repeat opacity-[0.20] dark:opacity-[0.12] dark:invert transition-all duration-700 mix-blend-multiply dark:mix-blend-screen" style={{ backgroundImage: "url('/1.jpg')", backgroundSize: "cover", backgroundPosition: "center 25%" }} />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-50 via-slate-50/60 to-transparent dark:from-[#09090b] dark:via-[#09090b]/60 dark:to-transparent" />
      </div>

      <nav className="relative z-40 w-full bg-white/70 dark:bg-[#09090b]/70 backdrop-blur-xl border-b border-slate-200 dark:border-white/5 px-6 py-4 flex justify-between items-center transition-colors duration-300">
        <div className="flex items-center gap-3 cursor-pointer group" onClick={() => setPhase("hero")}>
          <ManzilMark />
          <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">Manzil</span>
        </div>
        <div className="flex items-center gap-4 md:gap-6">
           <button onClick={() => setIsDarkMode(!isDarkMode)} className="p-2.5 text-slate-500 dark:text-slate-400 hover:text-amber-500 dark:hover:text-amber-400 transition-colors rounded-full bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 shadow-sm">
             {isDarkMode ? '☀️' : '🌙'}
           </button>
           
           {token ? (
             <div className="flex items-center gap-4 md:gap-6">
                <button 
                  onClick={() => {
                    setPhase("saved");
                    if (savedProps.length > 0) setActiveProperty(savedProps[0]);
                  }} 
                  title="Saved Properties"
                  className={`flex items-center gap-1.5 p-2 rounded-full transition-all ${phase === 'saved' ? 'text-amber-500 bg-amber-50 dark:bg-amber-500/10' : 'text-slate-500 dark:text-slate-400 hover:text-amber-500 hover:bg-slate-100 dark:hover:bg-white/5'}`}
                >
                  <BookmarkIcon filled={phase === 'saved'} />
                  {savedProps.length > 0 && (
                    <span className="text-xs font-bold">{savedProps.length}</span>
                  )}
                </button>
                <button onClick={handleLogout} className="px-6 py-2.5 text-sm font-bold bg-slate-200 dark:bg-white/10 hover:bg-slate-300 dark:hover:bg-white/20 text-slate-900 dark:text-white rounded-full transition-all">
                  Log Out
                </button>
             </div>
           ) : (
             <div className="flex items-center gap-4 md:gap-6">
               <button onClick={() => { setAuthMode("login"); setShowAuth(true); }} className="text-sm font-semibold text-slate-600 dark:text-slate-300 hover:text-amber-500 dark:hover:text-amber-400 transition-colors">Log In</button>
               <button onClick={() => { setAuthMode("signup"); setShowAuth(true); }} className="px-6 py-2.5 text-sm font-bold bg-amber-500 hover:bg-amber-600 text-slate-900 rounded-full transition-all shadow-md shadow-amber-500/20">
                 Sign Up
               </button>
             </div>
           )}
        </div>
      </nav>

      {/* --- AUTH MODAL --- */}
      {showAuth && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 dark:bg-black/80 backdrop-blur-sm">
          <div className="bg-white dark:bg-[#18181b] border border-slate-200 dark:border-white/10 p-8 md:p-10 rounded-3xl w-full max-w-md shadow-2xl relative animate-in fade-in zoom-in-95">
            <button onClick={() => setShowAuth(false)} className="absolute top-6 right-6 text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"><XIcon /></button>
            <div className="flex justify-center mb-6"><ManzilMark size={48} /></div>
            
            <h2 className="text-2xl font-bold text-center text-slate-900 dark:text-white mb-2">
              {authMode === "login" ? "Welcome Back" : "Create an Account"}
            </h2>
            <p className="text-slate-500 dark:text-slate-400 text-center text-sm mb-6">
              {authMode === "login" ? "Sign in to access your saved properties." : "Join Manzil to sync your property searches."}
            </p>
            
            {authError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-500 text-sm font-medium text-center">
                {authError}
              </div>
            )}
            
            <form onSubmit={handleAuthSubmit} className="space-y-4">
              <input 
                type="email" 
                placeholder="Email address" 
                value={authForm.email}
                onChange={e => setAuthForm({...authForm, email: e.target.value})}
                className="w-full bg-slate-50 dark:bg-black border border-slate-300 dark:border-white/10 rounded-xl px-5 py-3.5 text-slate-900 dark:text-white outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-all font-medium" 
                required
              />
              <input 
                type="password" 
                placeholder="Password" 
                value={authForm.password}
                onChange={e => setAuthForm({...authForm, password: e.target.value})}
                className="w-full bg-slate-50 dark:bg-black border border-slate-300 dark:border-white/10 rounded-xl px-5 py-3.5 text-slate-900 dark:text-white outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-all font-medium" 
                required
              />
              <button type="submit" className="w-full bg-amber-500 hover:bg-amber-600 text-slate-900 font-bold py-3.5 rounded-xl transition-colors shadow-lg shadow-amber-500/30">
                {authMode === "login" ? "Sign In" : "Sign Up"}
              </button>
            </form>

            <div className="mt-6 text-center">
              <button 
                onClick={() => { setAuthMode(authMode === "login" ? "signup" : "login"); setAuthError(""); }} 
                className="text-sm font-medium text-slate-500 hover:text-amber-500 dark:hover:text-amber-400 transition-colors"
              >
                {authMode === "login" ? "Don't have an account? Sign up" : "Already have an account? Log in"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- HERO PHASE --- */}
      {phase === "hero" && (
        <main className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-6 text-center animate-in fade-in duration-700">
          <div className="px-4 py-1.5 rounded-full border border-amber-200 dark:border-amber-500/30 bg-amber-50/80 dark:bg-amber-500/10 backdrop-blur-md text-amber-600 dark:text-amber-400 text-xs font-bold tracking-widest uppercase mb-6 shadow-sm">
            Intelligent Property Search
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-slate-900 dark:text-white tracking-tight leading-tight max-w-3xl mb-6">
            Find the perfect place, <br className="hidden md:block"/><span className="text-amber-500 dark:text-amber-400">instantly.</span>
          </h1>
          <p className="text-base md:text-lg text-slate-600 dark:text-slate-400 max-w-2xl mb-10 leading-relaxed">
            Skip the endless filters. Just tell our AI what you are looking for, and we'll map the best matches immediately.
          </p>
          <form onSubmit={handleSearch} className="w-full max-w-2xl relative">
            <div className={`relative flex flex-col md:flex-row items-center bg-white/90 dark:bg-[#18181b]/90 backdrop-blur-xl border-2 hover:border-amber-300 dark:hover:border-amber-500/50 rounded-2xl md:rounded-full p-2 shadow-xl shadow-slate-200/50 dark:shadow-none transition-all focus-within:border-amber-500 focus-within:shadow-2xl ${isRecording ? 'border-red-400 dark:border-red-500/50 shadow-red-500/20' : 'border-slate-200 dark:border-white/10'}`}>
              <div className="hidden md:block pl-5 text-amber-500 dark:text-amber-400"><SearchIcon /></div>
              <input 
                autoFocus value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., 5 marla near a school in DHA..."
                className="flex-1 w-full bg-transparent border-none outline-none px-5 py-3 text-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 font-medium"
              />
              
              {/* MIC BUTTON */}
              <button 
                type="button" 
                onClick={toggleRecording} 
                className={`p-3 mr-2 rounded-full transition-colors ${isRecording ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 shadow-inner' : 'text-slate-400 hover:text-amber-500 dark:hover:text-amber-400 hover:bg-slate-100 dark:hover:bg-white/5'}`}
                title={isRecording ? "Click to stop recording" : "Voice Search"}
              >
                <MicIcon isRecording={isRecording} />
              </button>

              <button type="submit" className="w-full md:w-auto mt-2 md:mt-0 bg-amber-500 hover:bg-amber-600 text-slate-900 px-8 py-3 rounded-xl md:rounded-full font-bold transition-colors shadow-md">Search</button>
            </div>
          </form>
          {errorMessage && <div className="mt-6 text-red-500 font-medium animate-in fade-in">⚠️ {errorMessage}</div>}
        </main>
      )}

      {/* --- CLARIFICATION PHASE --- */}
      {phase === "clarify" && clarification && (
        <main className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-6 text-center animate-in fade-in duration-700">
          <div className="w-16 h-16 bg-amber-100 dark:bg-amber-500/20 rounded-full flex items-center justify-center text-amber-500 mb-6 mx-auto">
            <SparkleIcon />
          </div>
          <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight mb-8">
            {clarification.question}
          </h2>
          <div className="flex flex-wrap justify-center gap-4 max-w-2xl mx-auto">
            {clarification.options
              .filter(opt => ["Commercial Plot", "Residential Plot"].includes(opt)) // <-- ADD THIS LINE
              .map((opt) => (
              <button
                key={opt}
                onClick={() => {
                  executeSearch(query, clarificationIntent, clarification.field, opt);
                }}
                className="px-6 py-3 bg-white dark:bg-[#18181b] border-2 border-slate-200 dark:border-white/10 hover:border-amber-500 dark:hover:border-amber-500 rounded-xl font-bold text-slate-700 dark:text-slate-200 transition-all shadow-sm hover:shadow-md"
              >
                {opt}
              </button>
            ))}
          </div>
        </main>
      )}

      {/* --- THINKING PHASE --- */}
      {phase === "thinking" && (
        <div className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-80px)] animate-in fade-in">
          <div className="flex gap-3 mb-6">
            <div className="w-3 h-3 rounded-full bg-amber-500 dark:bg-amber-400 animate-bounce"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500 dark:bg-amber-400 animate-bounce delay-100"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500 dark:bg-amber-400 animate-bounce delay-200"></div>
          </div>
          <p className="text-slate-600 dark:text-slate-400 font-bold tracking-widest uppercase text-sm bg-white/50 dark:bg-black/50 px-4 py-1 rounded-full backdrop-blur-md">Connecting to Core AI...</p>
        </div>
      )}

      {/* --- RESULTS / SAVED PHASE (SPLIT SCREEN) --- */}
      {(phase === "results" || phase === "saved") && (
        <div className="relative z-10 min-h-[calc(100vh-73px)] grid grid-cols-1 lg:grid-cols-12 animate-in slide-in-from-bottom-8 duration-700">
          <div className="lg:col-span-5 px-6 md:px-10 py-8 overflow-y-auto h-[calc(100vh-73px)] custom-scrollbar bg-white/80 dark:bg-[#09090b]/80 backdrop-blur-md">
            {phase === "results" ? (
              <div className="mb-8 p-5 rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-100 dark:border-amber-800/30">
                <p className="text-amber-800 dark:text-amber-200 font-medium text-sm leading-relaxed">✨ {summary}</p>
              </div>
            ) : (
              <div className="mb-8 p-5 rounded-xl bg-slate-100 dark:bg-white/5 border border-slate-200 dark:border-white/10">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">Your Saved Properties</h2>
                <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">Review the listings you've bookmarked.</p>
              </div>
            )}

            <div className="space-y-5 pb-20">
              {phase === "results" && results.length === 0 && <p className="text-center text-slate-500 py-10">No properties found matching your criteria.</p>}
              {phase === "saved" && savedProps.length === 0 && (
                <div className="text-center py-10">
                  <BookmarkIcon filled={false} className="mx-auto h-12 w-12 text-slate-300 dark:text-slate-700 mb-3" />
                  <p className="text-slate-500 font-medium">You haven't saved any properties yet.</p>
                </div>
              )}
              {(phase === "results" ? results : savedProps).map((r, index) => (
                <PropertyCard key={r.id || index} r={r} />
              ))}
            </div>
          </div>

          <div className="hidden lg:block lg:col-span-7 relative h-[calc(100vh-73px)] sticky top-[73px] border-l border-slate-200 dark:border-white/5">
            {activeProperty ? (
              <iframe title="Property Map" width="100%" height="100%" style={{ border: 0, filter: isDarkMode ? 'invert(90%) hue-rotate(180deg)' : 'none' }} loading="lazy" src={`https://maps.google.com/maps?q=${activeProperty.latitude},${activeProperty.longitude}&t=&z=15&ie=UTF8&iwloc=&output=embed`}></iframe>
            ) : (
              <div className="w-full h-full bg-slate-100 dark:bg-[#18181b] flex items-center justify-center"><p className="text-slate-500 font-medium">Select a property to view on map</p></div>
            )}
            {activeProperty && (
              <div className="absolute top-6 right-6 w-72 bg-white/95 dark:bg-[#18181b]/95 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-2xl p-5 shadow-2xl">
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100 dark:border-white/5">
                  <div className="flex items-center gap-2">
                    <TrendingUp />
                    <h4 className="font-bold text-slate-900 dark:text-white text-sm">Market Insights</h4>
                  </div>
                  <a href={`https://www.google.com/maps/search/?api=1&query=${activeProperty.latitude},${activeProperty.longitude}`} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-xs font-bold text-amber-600 dark:text-amber-400 hover:underline">
                    Open Maps <ExternalLinkIcon />
                  </a>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">Area Avg. Price</p>
                    <p className="text-base font-black text-slate-900 dark:text-white">1.95 - 2.45 Cr</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">1-Yr Trend</p>
                    <p className="text-base font-black text-emerald-600 dark:text-emerald-400">+12%</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}