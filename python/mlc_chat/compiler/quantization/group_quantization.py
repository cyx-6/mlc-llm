"""The group quantization schema"""
from typing import Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from . import QuantizeConfig
from tvm.relax.frontend import nn
from tvm import DataType, te, tir, DataTypeCode


@dataclass
class GroupQuantizeConfig:
    group_size: int
    quantize_dtype: DataType  # "int3", "int4", "int8"
    storage_dtype: DataType  # "uint32"
    weight_dtype: DataType  # "float16", "float32"
    symmetric: bool = True
    transpose: bool = False

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0

    def __post_init__(self):
        assert (
            isinstance(self.weight_dtype, DataType)
            and self.weight_dtype.type_code == DataTypeCode.FLOAT
        )
        assert (
            isinstance(self.quantize_dtype, DataType)
            and self.quantize_dtype.type_code == DataTypeCode.INT
        )
        assert (
            isinstance(self.storage_dtype, DataType)
            and self.storage_dtype.type_code == DataTypeCode.UINT
        )
        self.num_elem_per_storage = self.storage_dtype.bits // self.quantize_dtype.bits
        assert (
            self.num_elem_per_storage > 0
        ), "Storage unit should have more bits than single quantized elemtent"
        assert (
            self.group_size % self.num_elem_per_storage == 0
        ), "Group size should be divisible by numbers of elements per storage"
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        self.max_int_value = (1 << (self.quantize_dtype.bits - 1)) - 1


def te_quantize(weight: te.Tensor, config: GroupQuantizeConfig) -> Tuple[te.Tensor, te.Tensor]:
    assert len(weight.shape) == 2
    n, m = weight.shape
    # compute scale per group
    r = te.reduce_axis((0, config.group_size), name="r")
    num_group = tir.ceildiv(m, config.group_size)
    scale_shape = (n, num_group)
    max_abs = te.compute(
        shape=scale_shape,
        fcompute=lambda i, j: tir.max(
            tir.abs(weight[i, j * config.group_size + r]),
            tir.const(1e-4, config.weight_dtype),
            axis=r,
            where=j * config.group_size + r < m,
        ),
    )
    scale = tir.div(max_abs, config.max_int_value)

    # compute scaled weight
    tir_max_int = tir.const(config.max_int_value, config.weight_dtype)
    tir_zero = tir.const(0, config.weight_dtype)
    tir_max_int_2 = tir.const(config.max_int_value * 2, config.weight_dtype)
    scaled_weight = te.compute(
        shape=weight.shape,
        fcompute=lambda i, j: tir.min(
            tir.max(
                tir.round(weight[i, j] / scale[i, j // config.group_size] + tir_max_int), tir_zero
            ),
            tir_max_int_2,
        ).astype(config.storage_dtype),
    )

    # compute quantized weight per storage
    r = te.reduce_axis((0, config.num_elem_per_storage), name="r")
    num_storage = config.num_storage_per_group * num_group
    quantized_weight_shape = (n, num_storage)
    quantized_weight = te.compute(
        shape=quantized_weight_shape,
        fcompute=lambda i, j: tir.sum(
            scaled_weight[i, j * config.num_elem_per_storage + r]
            << (r * config.quantize_dtype.bits),
            axis=r,
            where=j * config.num_elem_per_storage + r < m,
        ),
    )
    return quantized_weight, scale


def te_dequantize(
    weight: te.Tensor,
    scale: te.Tensor,
    config: GroupQuantizeConfig,
    reshape: Optional[List[Union[int, tir.PrimExpr]]] = None,
) -> te.Tensor:
    if reshape is None:
        reshape = [weight.shape[0], weight.shape[1] * config.num_elem_per_storage]
    n, m = reshape
    tir_bin_mask = tir.const((1 << config.quantize_dtype.bits) - 1, config.storage_dtype)
    tir_max_int = tir.const(config.max_int_value, config.weight_dtype)
    dequantized_weight = te.compute(
        shape=reshape,
        fcompute=lambda i, j: tir.multiply(
            tir.subtract(
                tir.bitwise_and(
                    tir.shift_right(
                        weight[i, j // config.num_elem_per_storage],
                        (j % config.num_elem_per_storage) * config.storage_dtype.bits,
                        config.storage_dtype,
                    ),
                    tir_bin_mask,
                ),
                tir_max_int,
                config.weight_dtype,
            ),
            scale[i, j // config.group_size],
        ),
    )
    return dequantized_weight


class GroupQuantizeLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: GroupQuantizeConfig,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        n_group = tir.ceildiv(in_features, config.group_size)
        self.weight = nn.Parameter(
            (
                out_features,
                n_group * config.num_elem_per_storage,
            ),
            config.storage_dtype,
        )
        self.scale = nn.Parameter((out_features, n_group), config.weight_dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), config.weight_dtype)
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, config: GroupQuantizeConfig):
        return GroupQuantizeLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=config,
            bias=linear.bias,
            out_dtype=linear.out_dtype,
        )

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        w = nn.op.tensor_expr_op(
            lambda weight, scale: te_dequantize(
                weight, scale, self.config, [self.out_features, self.in_features]
            ),
            name_hint="group_dequantize",
            args=[self.weight, self.scale],
        )
        w = nn.op.permute_dims(w)
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x


class GroupQuantizeMutator(nn.Mutator):
    def __init__(self, config: GroupQuantizeConfig) -> None:
        super().__init__()
        self.config = config

    def visit_module(self, name: str, node: nn.Module) -> Any:
        if isinstance(node, nn.Linear) and node.weight.dtype == self.config.weight_dtype:
            return GroupQuantizeLinear.from_linear(node, self.config)
        return self.visit(name, node)
