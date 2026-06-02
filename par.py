pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu                             
                                                                                                             
  3. Sanity checks                                                                                             
                                                                                                             
  python -c "import QEfficient; print('import OK', QEfficient.__version__)"                                    
  python -c "import transformers; print('transformers', transformers.__version__)"                             
  python -c "                                                                                                  
  import inspect, torch                                                                                        
  print('torch', torch.__version__)                                                                          
  src = inspect.getsource(torch.export.exported_program.ExportedProgram.named_buffers)                         
  assert 'gm_state_dict' in src, 'torch build MISSING PR #182230'
  print('invoke_subgraph fix: OK')                                                                             
  "                                  
