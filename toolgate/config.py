from dataclasses import dataclass


@dataclass
class ToolGateConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"
    dtype: str = "float16"            # good for your 8GB RTX 5050
    tau: float = 0.5                  # the accuracy-vs-cost dial
    max_new_tokens: int = 256
    load_in_4bit: bool = False        # flip True to try a 3B model later
    direct_prefill: str = "I can answer this directly without using any tools. "
    tool_prefill: str = "I need to use a tool to answer this correctly. "