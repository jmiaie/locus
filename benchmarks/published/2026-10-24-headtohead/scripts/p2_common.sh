# sourced by phase-2 workers: no OpenRouter key in worker env; all LLM traffic goes through the local gateway
unset OPENROUTER_API_KEY
export OPENAI_API_KEY=sk-local-gateway-dummy
export MEM0_TELEMETRY=False POSTHOG_DISABLED=1 ANONYMIZED_TELEMETRY=False GRAPHITI_TELEMETRY_ENABLED=false DO_NOT_TRACK=1
export LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export LETTA_TELEMETRY_ENABLED=false
