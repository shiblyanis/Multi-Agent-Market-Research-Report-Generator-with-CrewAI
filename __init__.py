"""Multi-agent market research and report generator built on CrewAI."""
import os

# Must be set before crewai is imported anywhere: avoids interactive tracing
# prompts and outbound telemetry during runs.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
