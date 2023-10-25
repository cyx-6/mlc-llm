"""The base class of quantization schema"""
from dataclasses import dataclass
from typing import Any, Callable, Dict
from tvm.relax.frontend import nn
from tvm import DataType
from group_quantization import GroupQuantizeConfig, GroupQuantizeMutator


@dataclass
class QuantizeConfig:
    name: str
    kind: str
    mutator: nn.Mutator
    config: Any

    def apply(self, mod: nn.Module) -> nn.Module:
        return self.mutator(self.config)(mod)


QUANT: Dict[str, QuantizeConfig] = {
    "q4f16_1": QuantizeConfig(
        name="q4f16_1",
        kind="GroupQuantization",
        mutator=GroupQuantizeMutator,
        config=GroupQuantizeConfig(
            group_size=32,
            quantize_dtype=DataType("float16"),
            storage_dtype=DataType("int4"),
            symmetric=True,
            transpose=False,
        ),
    )
}
