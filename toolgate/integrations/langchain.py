"""One-line LangChain wrapper (the 'real framework' piece)."""
from ..gate import ToolGate


class ToolGateCallback:
    """Consult the probe BEFORE a LangChain agent decides to invoke tools."""
    def __init__(self, gate: ToolGate):
        self.gate = gate

    def should_use_tools(self, prompt: str) -> bool:
        return self.gate.probe.needs_tool(prompt)


def gated_agent(gate: ToolGate):
    """Decorator: short-circuits an agent when no tool is needed."""
    def decorator(agent_fn):
        def wrapper(prompt: str, *args, **kwargs):
            if not gate.probe.needs_tool(prompt):
                return gate.generate(prompt)["response"]
            return agent_fn(prompt, *args, **kwargs)
        return wrapper
    return decorator