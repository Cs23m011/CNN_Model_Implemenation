bufs = (
            list(candidate.buffers())
            if isinstance(candidate, torch.fx.GraphModule)
            else []
        )
        _probe_path = "none"
        if bufs and all(isinstance(buf, FunctionalTensor) for buf in bufs):
            gm = candidate
            _probe_path = "reuse"
        if gm is None:
            gm = materialize_as_graph(
                subgraph, operands, subgraph_decomp_table=subgraph_decomp_table
            )
            _probe_path = "materialize"

        # --- PROBE (temporary) ---
        import collections as _c
        _nph = sum(1 for _n in gm.graph.nodes if _n.op == "placeholder")
        _nop = len(operands)
        if _nph != _nop:
            _shapes = [tuple(getattr(o, "shape", ())) for o in operands]
            _dupes = [s for s, cnt in _c.Counter(_shapes).items() if cnt > 1 and s]
            print(f"[PROBE] path={_probe_path} nbufs={len(bufs)} "
                  f"operands={_nop} placeholders={_nph} MISMATCH dupes={_dupes}")
        else:
            print(f"[PROBE] path={_probe_path} nbufs={len(bufs)} "
                  f"operands={_nop} placeholders={_nph} OK")
        # --- END PROBE ---

        schema_gen = HopSchemaGenerator(self)
