# Line 41 currently:
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

# Add immediately after it:
from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc3D, CtxScatterFunc3D
# Add after the imports block, before line 44 (QEffMixtralRotaryEmbedding):
EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", "16"))
class QEffMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    """
    Decode-optimised MoE block. Uses pre-stacked weight tensors and
    torch.bmm to process only the top-k selected experts per token —
    no loop over all experts, no masking waste.
    """

    def __qeff_init__(self):
        self.gate_proj_w = []
        self.up_proj_w = []
        self.down_proj_w = []
        with torch.no_grad():
            for e in range(self.num_experts):
                self.gate_proj_w.append(self.experts[e].w1.weight.T)  # [H, I]
                self.up_proj_w.append(self.experts[e].w3.weight.T)    # [H, I]
                self.down_proj_w.append(self.experts[e].w2.weight.T)  # [I, H]
            self.gate_proj_w = torch.stack(self.gate_proj_w)  # [E, H, I]
            self.up_proj_w   = torch.stack(self.up_proj_w)    # [E, H, I]
            self.down_proj_w = torch.stack(self.down_proj_w)  # [E, I, H]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        T = batch_size * sequence_length
        hidden_states = hidden_states.view(T, hidden_dim)

        router_logits = self.gate(hidden_states)                              # [T, E]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_w, top_i = torch.topk(routing_weights, self.top_k, dim=-1)       # [T, K]
        top_w = top_w / top_w.sum(-1, keepdim=True)                          # normalize
        top_w = top_w.to(hidden_states.dtype)

        # Gather weights only for the selected experts — shape [T*K, H, I]
        gate_proj_w = self.gate_proj_w[top_i.flatten()]
        up_proj_w   = self.up_proj_w[top_i.flatten()]
        down_proj_w = self.down_proj_w[top_i.flatten()]

        # Expand each token's hidden state for its top-k experts — [T*K, 1, H]
        expert_in = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, hidden_dim)

        gate        = torch.bmm(expert_in, gate_proj_w)               # [T*K, 1, I]
        up          = torch.bmm(expert_in, up_proj_w)                  # [T*K, 1, I]
        experts_out = torch.bmm(up * self.experts[0].act_fn(gate), down_proj_w)  # [T*K, 1, H]

        experts_out = experts_out.view(T, self.top_k, hidden_dim)
        experts_out = experts_out * top_w.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)                           # [T, H]

        return experts_out.view(batch_size, sequence_length, hidden_dim), router_logits


def _ctx_scatter_gather_mixtral_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    act_fn,
    T: int,
) -> torch.Tensor:
    """NSP-blocked expert helper for Mixtral prefill dispatch."""
    batch_size, hidden_size = T2Ei.shape[0], x.shape[1]
    scatter_idx = (torch.cumsum(T2Ei.long(), dim=1) - 1).to(torch.int32)
    invalid_mask = ~T2Ei
    INT32_MAX = torch.tensor(torch.iinfo(torch.int32).max, dtype=torch.int32, device=x.device)
    scatter_safe_idx = torch.where(invalid_mask, INT32_MAX, scatter_idx)

    x_prime = torch.zeros(batch_size, T, hidden_size, dtype=x.dtype, device=x.device)
    x_prime = CtxScatterFunc3D.apply(x_prime, scatter_safe_idx, x.unsqueeze(0).expand(batch_size, -1, -1))

    gate_prime = x_prime @ W_g
    up_prime   = x_prime @ W_u
    down_prime = (up_prime * act_fn(gate_prime)) @ W_d

    valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
    row_range  = torch.arange(T, device=x.device, dtype=torch.int32).unsqueeze(0)
    down_prime = torch.where((row_range < valid_rows).unsqueeze(-1), down_prime, torch.zeros_like(down_prime))

    gather_idx = torch.where(invalid_mask, INT32_MAX, scatter_idx)
    delta_out  = CtxGatherFunc3D.apply(down_prime, gather_idx)
    delta_out  = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(delta_out), delta_out)
    return delta_out


class QEffPrefillChunkedMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    """
    Prefill-optimised MoE block using NSP-blocked scatter/gather dispatch.
    Only activated during prefill export (prefill_only=True, enable_chunking=True).
    """

    def __qeff_init__(self):
        self.gate_proj_w = []
        self.up_proj_w = []
        self.down_proj_w = []
        with torch.no_grad():
            for e in range(self.num_experts):
                self.gate_proj_w.append(self.experts[e].w1.weight.T)
                self.up_proj_w.append(self.experts[e].w3.weight.T)
                self.down_proj_w.append(self.experts[e].w2.weight.T)
            self.gate_proj_w = torch.stack(self.gate_proj_w)  # [E, H, I]
            self.up_proj_w   = torch.stack(self.up_proj_w)    # [E, H, I]
            self.down_proj_w = torch.stack(self.down_proj_w)  # [E, I, H]

    def _forward_expert_blocked(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        num_nsp = EXPERT_BLOCKING_NUM_NSP
        if self.num_experts % num_nsp != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by EXPERT_BLOCKING_NUM_NSP ({num_nsp})"
            )
        local_experts = self.num_experts // num_nsp

        rw  = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
        W_g = self.gate_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_u = self.up_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_d = self.down_proj_w.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()

        expert_out_partial = x.new_zeros((num_nsp, T, H))
        for slot in range(local_experts):
            routing_weight = rw[:, slot, :].unsqueeze(-1)       # [num_nsp, T, 1]
            T2Ei  = routing_weight.squeeze(-1) > 0               # [num_nsp, T]
            delta = _ctx_scatter_gather_mixtral_expert_blocked(
                x=x,
                T2Ei=T2Ei,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                act_fn=self.experts[0].act_fn,
                T=T,
            )
            expert_out_partial = expert_out_partial + (delta * routing_weight)

        return expert_out_partial.sum(dim=0)  # [T, H]

    def orig_forward(self, hidden_states: torch.Tensor) -> tuple:
        """Serial per-expert loop — kept for parity testing."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        T = batch_size * sequence_length
        x = hidden_states.view(T, hidden_dim)

        router_logits    = self.gate(x)
        routing_weights  = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_w, top_i     = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w            = top_w / top_w.sum(-1, keepdim=True)
        top_w            = top_w.to(hidden_states.dtype)
        routing_weights  = torch.zeros_like(router_logits)
        routing_weights.scatter_(1, top_i, top_w)

        expert_out = x.new_zeros((T, hidden_dim))
        for e in range(self.num_experts):
            rw  = routing_weights[:, e].unsqueeze(-1)
            W_g = self.experts[e].w1.weight.T
            W_u = self.experts[e].w3.weight.T
            W_d = self.experts[e].w2.weight.T
            gate = x @ W_g
            up   = x @ W_u
            down = (up * self.experts[e].act_fn(gate)) @ W_d
            expert_out += down * rw
        return expert_out.view(batch_size, sequence_length, hidden_dim), router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        T = batch_size * sequence_length
        x = hidden_states.view(T, hidden_dim)

        router_logits    = self.gate(x)
        routing_weights  = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_w, top_i     = torch.topk(routing_weights, self.top_k, dim=-1)
        top_w            = top_w / top_w.sum(-1, keepdim=True)
        top_w            = top_w.to(hidden_states.dtype)
        routing_weights  = torch.zeros_like(router_logits)
        routing_weights.scatter_(1, top_i, top_w)

        if self.num_experts % EXPERT_BLOCKING_NUM_NSP == 0:
            expert_out = self._forward_expert_blocked(x=x, routing_weights=routing_weights)
            return expert_out.view(batch_size, sequence_length, hidden_dim), router_logits

        # Fallback: serial loop if num_experts not divisible by NSP
        expert_out = x.new_zeros((T, hidden_dim))
        for e in range(self.num_experts):
            rw  = routing_weights[:, e].unsqueeze(-1)
            W_g = self.experts[e].w1.weight.T
            W_u = self.experts[e].w3.weight.T
            W_d = self.experts[e].w2.weight.T
            gate = x @ W_g
            up   = x @ W_u
            down = (up * self.experts[e].act_fn(gate)) @ W_d
            expert_out += down * rw
        return expert_out.view(batch_size, sequence_length, hidden_dim), router_logits
