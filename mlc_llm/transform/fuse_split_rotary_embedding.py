from tvm import relax
from tvm.script import tir as T
from tvm.relax.dpl import (
    PatternContext,
    is_op,
    rewrite_bindings,
    wildcard,
    is_tuple_get_item,
    GlobalVarPattern,
    TuplePattern,
    is_shape,
)
from tvm.script import relax as R


@T.prim_func
def split_rotary_7b(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(12288)), "float16"),
    cos: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    sin: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    T_split: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    T_split_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    T_split_2: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    n: T.int64,
):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_split"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[v_ax0, v_ax1, v_ax2],
                A[v_ax0, v_ax1, v_ax2 + T.int64(4096)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(8192)],
            )
            T.writes(
                T_split[v_ax0, v_ax1, v_ax2],
                T_split_1[v_ax0, v_ax1, v_ax2],
                T_split_2[v_ax0, v_ax1, v_ax2],
            )
            T_split[v_ax0, v_ax1, v_ax2] = cos[n - T.int64(1), v_ax2 % 128] * A[
                v_ax0, v_ax1, v_ax2
            ] + sin[n - T.int64(1), v_ax2 % 128] * T.Select(
                T.int64(64) <= v_ax2 % 128,
                A[v_ax0, v_ax1, v_ax2 - T.int64(64)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(64)] * T.float16(-1),
            )
            T_split_1[v_ax0, v_ax1, v_ax2] = cos[n - T.int64(1), v_ax2 % 128] * A[
                v_ax0, v_ax1, v_ax2 + T.int64(4096)
            ] + sin[n - T.int64(1), v_ax2 % 128] * T.Select(
                T.int64(64) <= v_ax2 % 128,
                A[
                    v_ax0,
                    v_ax1,
                    v_ax2 + T.int64(4096) - T.int64(64),
                ],
                A[
                    v_ax0,
                    v_ax1,
                    v_ax2 + T.int64(4096) + T.int64(64),
                ]
                * T.float16(-1),
            )
            T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(8192)]


@T.prim_func
def split_rotary_13b(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(15360)), "float16"),
    cos: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    sin: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    T_split: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16"),
    T_split_1: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16"),
    T_split_2: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16"),
    n: T.int64,
):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(5120)):
        with T.block("T_split"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[v_ax0, v_ax1, v_ax2],
                A[v_ax0, v_ax1, v_ax2 + T.int64(5120)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(10240)],
            )
            T.writes(
                T_split[v_ax0, v_ax1, v_ax2],
                T_split_1[v_ax0, v_ax1, v_ax2],
                T_split_2[v_ax0, v_ax1, v_ax2],
            )
            T_split[v_ax0, v_ax1, v_ax2] = cos[n - T.int64(1), v_ax2 % 128] * A[
                v_ax0, v_ax1, v_ax2
            ] + sin[n - T.int64(1), v_ax2 % 128] * T.Select(
                T.int64(64) <= v_ax2 % 128,
                A[v_ax0, v_ax1, v_ax2 - T.int64(64)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(64)] * T.float16(-1),
            )
            T_split_1[v_ax0, v_ax1, v_ax2] = cos[n - T.int64(1), v_ax2 % 128] * A[
                v_ax0, v_ax1, v_ax2 + T.int64(5120)
            ] + sin[n - T.int64(1), v_ax2 % 128] * T.Select(
                T.int64(64) <= v_ax2 % 128,
                A[
                    v_ax0,
                    v_ax1,
                    v_ax2 + T.int64(5120) - T.int64(64),
                ],
                A[
                    v_ax0,
                    v_ax1,
                    v_ax2 + T.int64(5120) + T.int64(64),
                ]
                * T.float16(-1),
            )
            T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(10240)]


def fuse_split_rotary_embedding(mod, num_layers):
    func_map = {32: split_rotary_7b, 40: split_rotary_13b}

    if num_layers not in func_map:
        return mod

    mod["split_rotary"] = func_map[num_layers]
    gvar = mod.get_global_var("split_rotary")
    relax.expr._update_struct_info(
        gvar, mod.get_global_var("rotary_embedding1").struct_info
    )

    with PatternContext() as ctx:
        # lv3: R.Tuple(R.Tensor((1, 1, 4096), dtype="float16"), R.Tensor((1, 1, 4096), dtype="float16"), R.Tensor((1, 1, 4096), dtype="float16")) = R.split(lv2, indices_or_sections=[4096, 8192], axis=2)

        # lv1521: R.Tensor((1, 1, 4096), dtype="float16") = lv3[0]
        # lv1522: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv1521, R.shape([1, 1, 32, 128]))
        # lv1524: R.Tensor((1, 1, 4096), dtype="float16") = lv3[1]
        # lv1525: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv1524, R.shape([1, 1, 32, 128]))
        # lv1527: R.Tensor((1, 1, 4096), dtype="float16") = lv3[2]
        # lv1528: R.Tensor((1, 1, 32, 128), dtype="float16") = R.reshape(lv1527, R.shape([1, 1, 32, 128]))
        # lv1530 = R.call_tir(cls.rotary_embedding1, (lv1525, cos_cached1, sin_cached1), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float16"), tir_vars=R.shape([n]))
        # lv_1 = R.call_tir(cls.rotary_embedding1, (lv1522, cos_cached1, sin_cached1), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float16"), tir_vars=R.shape(

        inp_pat = wildcard()
        cos_cached = wildcard()
        sin_cached = wildcard()
        offset = wildcard()

        lv3 = is_op("relax.split")(inp_pat)
        lv1521 = is_tuple_get_item(lv3, 0)
        lv1522 = is_op("relax.reshape")(
            lv1521, is_shape([1, 1, num_layers, 128]), add_constraint=False
        )
        lv1521.used_by(lv1522)
        lv1524 = is_tuple_get_item(lv3, 1)
        lv1525 = is_op("relax.reshape")(
            lv1524, is_shape([1, 1, num_layers, 128]), add_constraint=False
        )
        lv1524.used_by(lv1525)
        lv1527 = is_tuple_get_item(lv3, 2)
        V = is_op("relax.reshape")(
            lv1527, is_shape([1, 1, num_layers, 128]), add_constraint=False
        )
        lv1527.used_by(V)

        Q = is_op("relax.call_tir")(
            GlobalVarPattern(),
            TuplePattern([lv1522, cos_cached, sin_cached]),
            offset,
            add_constraint=False,
        )
        K = is_op("relax.call_tir")(
            GlobalVarPattern(),
            TuplePattern([lv1525, cos_cached, sin_cached]),
            offset,
            add_constraint=False,
        )

        lv3.used_by(lv1521)
        lv3.used_by(lv1524)
        lv3.used_by(lv1527)
        lv1522.used_by(Q)
        lv1525.used_by(K)
        cos_cached.used_by(Q)
        sin_cached.used_by(Q)

    def rewriter(matchings, bindings):
        inp = matchings[inp_pat]
        cos = matchings[cos_cached]
        sin = matchings[sin_cached]
        call_tir = matchings[Q]
        n = bindings[call_tir].args[-1]
        out_sinfo = [
            R.Tensor((1, 1, num_layers * 128), dtype="float16"),
            R.Tensor((1, 1, num_layers * 128), dtype="float16"),
            R.Tensor((1, 1, num_layers * 128), dtype="float16"),
        ]
        lv3_new = R.call_tir(
            mod.get_global_var("split_rotary"),
            (inp, cos, sin),
            out_sinfo=out_sinfo,
            tir_vars=n,
        )
        lv1521_new = lv3_new[0]
        lv1522_new = R.reshape(lv1521_new, R.shape([1, 1, num_layers, 128]))
        lv1524_new = lv3_new[1]
        lv1525_new = R.reshape(lv1524_new, R.shape([1, 1, num_layers, 128]))
        lv1527_new = lv3_new[2]
        lv1528_new = R.reshape(lv1527_new, R.shape([1, 1, num_layers, 128]))

        return {
            matchings[lv3]: lv3_new,
            matchings[lv1521]: lv1521_new,
            matchings[lv1522]: lv1522_new,
            matchings[lv1524]: lv1524_new,
            matchings[lv1525]: lv1525_new,
            matchings[lv1527]: lv1527_new,
            matchings[V]: lv1528_new,
            matchings[Q]: lv1522_new,
            matchings[K]: lv1525_new,
        }

    mod["decode"] = rewrite_bindings(ctx, rewriter, mod["decode"])
    return mod