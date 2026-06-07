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
No [PROBE] lines were printed at all. Here's what's happening:

  The probe code was never reached. The crash occurs earlier, during torch.export (step 1/3 of ONNX export), with:

  TypeError: forward() missing 1 required positional argument: 'arg26_1'

  The probe is in gen_schema(), but the call stack shows the failure happens inside invoke_subgraph.__call__() → super().__call__() →
  dispatches to something that calls forward() with the wrong number of args. gen_schema is a separate code path that never gets invoked
  before the crash.

  The actual bug: forward() is called with 25 args but expects 26. This is the operand/placeholder mismatch your probe was designed to catch
  — but the crash happens before gen_schema is called, during the torch.export trace itself.

  To make the probe fire, you'd need to add it earlier — either inside __call__ before super().__call__(), or in the dispatch handler that
  ultimately calls forward(). The mismatch is already manifesting as a TypeError crash at the call site.
