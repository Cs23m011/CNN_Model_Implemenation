grep -nE "arg26_1|arg25_1|arg27_1" arg26.log | head -30
grep -nE "size=\(4096|size=\[4096|requires_grad=True.*4096|4096.*requires_grad=True" arg26.log | head -20
grep -nE "missing.*argument|positional argument|expected.*arg|forward\(\)" arg26.log | head -20
grep -nE "set_proxy_slot Parameter|set_proxy_slot.*Buffer|get_attr|_tensor_constant" arg26.log | head -40
