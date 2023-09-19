import tvm
from tvm import IRModule, relax, tir
from tvm.relax.dpl.pattern import is_op, is_tuple, wildcard


def check_matmul(ctx: relax.transform.PatternCheckContext, i: int) -> bool:
    a = ctx.annotated_expr[f"lora_{i}_a"]
    b = ctx.annotated_expr[f"lora_{i}_b"]
    return (
        isinstance(a, relax.TupleGetItem)
        and isinstance(b, relax.TupleGetItem)
        and len(a.struct_info.shape) == 2
        and len(b.struct_info.shape) == 2
        and isinstance(a.struct_info.shape[1], tir.Var)
        and isinstance(b.struct_info.shape[0], tir.Var)
        and a.struct_info.shape[1].same_as(b.struct_info.shape[0])
    )


def pattern_check(n: int):
    def f_pattern_check(ctx: relax.transform.PatternCheckContext) -> bool:
        for i in range(n):
            if not check_matmul(ctx, i):
                return False
        return True

    return f_pattern_check


def lora_matmul_pattern(n: int):
    assert n > 0
    concat_matmul = []
    annotations = {}
    for i in range(n):
        annotations[f"lora_{i}_a"] = a = wildcard()
        annotations[f"lora_{i}_b"] = b = wildcard()
        # annotations[f"lora_{i}_matmul"] = matmul = is_op("relax.matmul")(a, b)
        concat_matmul.append(is_op("relax.matmul")(a, b))
    if n == 1:
        return concat_matmul[0], annotations, pattern_check(n)
    concat = is_op("relax.concat")(is_tuple(concat_matmul))
    # annotations["concat"] = concat
    return concat, annotations, pattern_check(n)


@tvm.transform.module_pass(opt_level=0, name="FuseLoRAMatmul")
class FuseLoRAMatmul:
    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        for n in [3, 2, 1]:
            mod = relax.transform.FuseOpsByPattern([("lora_matmul", *lora_matmul_pattern(n))])(mod)

        return mod
