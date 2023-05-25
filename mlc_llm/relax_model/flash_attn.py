import tvm.tir as tir
from tvm.script import tir as T


def flash_attn_raw(
    H: int,
    D: int,
    is_decoder: bool = False,
    dtype: str = "float16",
):
    def if_decoder(x, y):
        if is_decoder:
            return x
        else:
            return y

    sm_scale = 1.0 / (float(D) ** 0.5)
    eps = 1e-5
    H = T.int64(H)
    D = T.int64(H)

    @T.prim_func
    def func(q: T.handle, k: T.handle, v: T.handle, o: T.handle) -> None:
        T.func_attr(
            {"op_pattern": 8, "tir.noalias": T.bool(True), "tir.is_scheduled": 1}
        )
        M = T.int64()
        N = T.int64()
        Q = T.match_buffer(q, (M, H, D), dtype)
        K = T.match_buffer(k, (N, H, D), dtype)
        V = T.match_buffer(v, (N, H, D), dtype)
        O = T.match_buffer(o, (M, H, D), dtype)

        m_now = T.alloc_buffer((H, M), dtype="float32", scope="local")
        m_prev = T.alloc_buffer((H, M), dtype="float32", scope="local")
        d_now = T.alloc_buffer((H, M), dtype="float32", scope="local")
        d_prev = T.alloc_buffer((H, M), dtype="float32", scope="local")
        Q_shared = T.alloc_buffer((H, M, D), dtype=dtype, scope="shared")
        O_shared = T.alloc_buffer((H, M, D), dtype="float32", scope="shared")

        for i, h in T.grid(M, H):
            for k in T.grid(D):
                with T.block("Q_shared"):
                    vi, vh, vk = T.axis.remap("SSS", [i, h, k])
                    Q_shared[vh, vi, vk] = Q[vi, vh, vk]

            for jo in T.grid((N + 31) // 32):
                with T.block("flashattn"):
                    vi, vh, vjo = T.axis.remap("SSR", [i, h, jo])

                    m_local = T.alloc_buffer(
                        ((T.int64(1)),), dtype="float32", scope="local"
                    )
                    a_sum_local = T.alloc_buffer(
                        ((T.int64(1)),), dtype="float32", scope="local"
                    )
                    o = T.alloc_buffer((D,), dtype="float32", scope="local")
                    K_shared_pad = T.alloc_buffer(
                        (T.int64(32), D), dtype=dtype, scope="shared"
                    )
                    V_shared_pad = T.alloc_buffer(
                        (T.int64(32), D), dtype=dtype, scope="shared"
                    )
                    X = T.alloc_buffer((T.int64(32),), dtype="float32", scope="shared")
                    A = T.alloc_buffer((T.int64(32),), dtype="float32", scope="shared")
                    X_mask = T.alloc_buffer(
                        (T.int64(32),), dtype="float32", scope="shared"
                    )

                    with T.init():
                        m_now[vh, vi] = T.min_value("float32")
                        m_prev[vh, vi] = T.min_value("float32")
                        d_now[vh, vi] = T.float32(0)
                        d_prev[vh, vi] = T.float32(0)
                        for k in T.grid(D):
                            with T.block("flashattn_O_init"):
                                vk = T.axis.spatial(D, k)
                                O_shared[vh, vi, vk] = T.float32(0)

                    for ji, k in T.grid(T.int64(32), D):
                        with T.block("K_shared_pad"):
                            vji, vk = T.axis.remap("SS", [ji, k])
                            K_shared_pad[vji, vk] = T.if_then_else(
                                32 * vjo + vji < N,
                                K[32 * vjo + vji, vh, vk],
                                T.cast(0, dtype),
                            )

                    for ji, k in T.grid(T.int64(32), D):
                        with T.block("V_shared_pad"):
                            vji, vk = T.axis.remap("SS", [ji, k])
                            V_shared_pad[vji, vk] = T.if_then_else(
                                32 * vjo + vji < N,
                                V[32 * vjo + vji, vh, vk],
                                T.cast(0, dtype),
                            )

                    for ji, k in T.grid(T.int64(32), D):
                        with T.block("flashattn_x"):
                            vji, vk = T.axis.remap("SR", (ji, k))
                            with T.init():
                                X[vji] = T.float32(0)
                            X[vji] = (
                                X[vji]
                                + T.cast(Q_shared[vh, vi, vk], "float32")
                                * T.cast(
                                    K_shared_pad[vji, vk],
                                    "float32",
                                )
                                * sm_scale
                            )

                    for ji in T.grid(T.int64(32)):
                        with T.block("flashattn_x_mask"):
                            vji = T.axis.spatial(T.int64(32), ji)
                            X_mask[vji] = X[vji] + T.if_then_else(
                                if_decoder(
                                    (vjo * 32 + vji >= N)
                                    or (M - vi > N - vjo * 32 - vji),
                                    vjo * 32 + vji >= N,
                                ),
                                T.min_value("float32"),
                                T.float32(0),
                            )

                    for ji in T.grid(T.int64(32)):
                        with T.block("flashattn_m_local"):
                            vji = T.axis.reduce(N, ji)
                            with T.init():
                                m_local[0] = T.min_value("float32")
                            m_local[0] = T.max(m_local[0], X_mask[vji])

                    with T.block("flashattn_m"):
                        m_prev[vh, vi] = m_now[vh, vi]
                        m_now[vh, vi] = T.max(m_prev[vh, vi], m_local[0])

                    for ji in T.grid(T.int64(32)):
                        with T.block("flashattn_a"):
                            vji = T.axis.spatial(N, ji)
                            A[vji] = T.exp(X_mask[vji] - m_now[vh, vi])

                    for ji in T.grid(T.int64(32)):
                        with T.block("flashattn_a_sum_local"):
                            vji = T.axis.reduce(N, ji)
                            with T.init():
                                a_sum_local[0] = T.float32(0)
                            a_sum_local[0] = a_sum_local[0] + A[vji]

                    with T.block("flashattn_d"):
                        d_prev[vh, vi] = d_now[vh, vi]
                        d_now[vh, vi] = d_prev[vh, vi] * T.exp(
                            m_prev[vh, vi] - m_now[vh, vi]
                        ) + T.max(a_sum_local[0], eps)

                    for ji, k in T.grid(T.int64(32), D):
                        with T.block("flashattn_o"):
                            vji = T.axis.reduce(N, ji)
                            vk = T.axis.spatial(D, k)
                            with T.init():
                                o[vk] = T.float32(0)
                            o[vk] = o[vk] + A[vji] * V_shared_pad[vji, vk]

                    for k in T.grid(D):
                        with T.block("flashattn_O_shared"):
                            vk = T.axis.spatial(D, k)
                            O_shared[vh, vi, vk] = (
                                O_shared[vh, vi, vk]
                                * d_prev[vh, vi]
                                * T.exp(m_prev[vh, vi] - m_now[vh, vi])
                                + o[vk]
                            ) / d_now[vh, vi]

            for k in T.grid(D):
                with T.block("flashattn_O"):
                    vi, vh, vk = T.axis.remap("SSS", [i, h, k])
                    O[vi, vh, vk] = O_shared[vh, vi, vk]

    return func


def flash_attn_gen(
    H: int,
    D: int,
    is_decoder: bool = False,
    dtype: str = "float16",
):
    func = flash_attn_raw(H, D, is_decoder, dtype)
    sch = tir.Schedule(func, enable_check=False)

    outer = sch.get_block("flashattn")
    i, h, jo = sch.get_loops(outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(h, "blockIdx.y")
    sch.decompose_reduction(outer, jo)

    def bind_tx_ty(axis):
        ty, tx = sch.split(axis, [4, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    Q_shared = sch.get_block("Q_shared")
    k = sch.get_loops(Q_shared)[-1]
    bind_tx_ty(k)

    O_init = sch.get_block("flashattn_O_init")
    k = sch.get_loops(O_init)[-1]
    bind_tx_ty(k)

    K_shared_pad = sch.get_block("K_shared_pad")
    j, k = sch.get_loops(K_shared_pad)[-2:]
    sch.unroll(j)
    sch.annotate(j, "pragma_unroll_explicit", 0)
    bind_tx_ty(k)

    V_shared_pad = sch.get_block("V_shared_pad")
    j, k = sch.get_loops(V_shared_pad)[-2:]
    sch.unroll(j)
    sch.annotate(j, "pragma_unroll_explicit", 0)
    bind_tx_ty(k)

    logits = sch.get_block("flashattn_x")
    j, k = sch.get_loops(logits)[-2:]
    jo, ji = sch.split(j, [4, None])
    ko, ki = sch.split(k, [None, 32])
    sch.unroll(ji)
    sch.annotate(ji, "pragma_unroll_explicit", 0)
    sch.unroll(ko)
    sch.annotate(ko, "pragma_unroll_explicit", 0)
    sch.bind(jo, "threadIdx.y")
    sch.bind(ki, "threadIdx.x")

    mask = sch.get_block("flashattn_x_mask")
    j = sch.get_loops(mask)[-1]
    jo, ji = sch.split(j, [4, 32])
    sch.bind(jo, "threadIdx.y")
    sch.bind(ji, "threadIdx.x")

    m_local = sch.get_block("flashattn_m_local")
    j = sch.get_loops(m_local)[-1]
    sch.bind(j, "threadIdx.x")

    a = sch.get_block("flashattn_a")
    j = sch.get_loops(a)[-1]
    sch.bind(j, "threadIdx.x")

    a_sum_local = sch.get_block("flashattn_a_sum_local")
    j = sch.get_loops(a_sum_local)[-1]
    sch.bind(j, "threadIdx.x")

    o = sch.get_block("flashattn_o")
    j, k = sch.get_loops(o)[-2:]
    sch.unroll(j)
    sch.annotate(j, "pragma_unroll_explicit", 0)
    bind_tx_ty(k)

    O_shared = sch.get_block("flashattn_O_shared")
    k = sch.get_loops(O_shared)[-1]
    bind_tx_ty(k)

    O = sch.get_block("flashattn_O")
    k = sch.get_loops(O)[-1]
    bind_tx_ty(k)

    return sch.mod["main"]
