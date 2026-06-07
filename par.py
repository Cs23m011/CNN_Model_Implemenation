gm: torch.fx.GraphModule = materialize_as_graph(
            subgraph, operands, subgraph_decomp_table=subgraph_decomp_table
        )
        # --- PROBE ---
        import collections
        _nph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
        if _nph != len(operands):
            _shapes = [tuple(getattr(o, "shape", ())) for o in operands]
            _dupes = [s for s, c in collections.Counter(_shapes).items() if c > 1 and s]
            print(f"[PROBE] operands={len(operands)} placeholders={_nph} MISMATCH; duplicate shapes={_dupes}")
        # --- END PROBE ---
