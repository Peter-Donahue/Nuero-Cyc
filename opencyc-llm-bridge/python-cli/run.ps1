$env:CYC_BRIDGE_BASE_URL = "http://localhost:8081"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.2"
$env:OLLAMA_TEMPERATURE = "0"
$env:CYC_SESSION_GENL_MT = '#$InferencePSC'
$env:LLM_TRACE_DIR = "traces"
python -m cyc_llm_cli