"""STEP 6-1 이미지별 attention output 기여: 추출(무개입 실행)과 개입(o_proj 뒤, residual 덧셈 앞).

정의 (층 ℓ, query 토큰 t, key 묶음 g):
  z_g[t] = concat_h Σ_{j∈g} A[h,t,j] v[h,j]     o_proj 입력 공간. head 순서와 GQA 대응은 모델과 같다 (repeat_kv).
  c_g[t] = W_O z_g[t]                             o_proj 이후 공간, bias 제외
  o[t]   = Σ_g c_g[t] + b_O                       b_O는 전체 재구성에서 한 번만 더한다
                                                  (Qwen2.5-VL 텍스트 o_proj는 bias가 없어 b_O = 0)
묶음: 이미지마다 image-pad 토큰 [start, end) 하나씩. 나머지 key 전부(시스템·질문 텍스트, vision_start/end 표시 토큰
포함)는 'nonimage' 하나로 묶는다. 묶음들은 key 위치 [0, T)를 겹침 없이 모두 나눈다 (검사).
A는 모델과 같은 입력으로 다시 계산한다.
- 같은 입력: 그 층 attention의 입력 hidden, 같은 q/k/v 투영(모델 dtype, bias 포함), 같은 mrope cos/sin,
  같은 mask(4D 가산·bool·없음 = 인과), 같은 scaling
- softmax와 가중합은 compute_dtype(기본 float32)로 한다.
범위: batch 1, KV cache 없는 전체 prefill (query·key 길이 = 입력 길이).
STEP 5-3B의 빠른 경로(prefix cache + suffix)는 아직 지원하지 않는다.
"""
import torch

from src.core.feature_validation import require


def attention_modules(model, layers):
    decoder = model.get_decoder()
    return {index: decoder.layers[index].self_attn for index in layers}


def question_positions(spans, length):
    """STEP 5-3B와 같은 질문 구간: 마지막 이미지의 vision_end 다음 토큰부터 마지막 입력 토큰까지."""
    start = spans[-1]["end"] + 1
    require(start < length, "No text token after the last image")
    return list(range(start, length))


def key_groups(length, spans):
    """[(이름, key 위치)]: image_0..image_{k-1} (image-pad [start, end)), nonimage (나머지 전부)."""
    owner = torch.full((length,), -1, dtype=torch.long)
    for index, span in enumerate(spans):
        require(0 <= span["start"] < span["end"] <= length, f"Invalid image span {span}")
        require(bool((owner[span["start"]:span["end"]] == -1).all()), "Image spans overlap")
        owner[span["start"]:span["end"]] = index
    groups = [(f"image_{i}", (owner == i).nonzero(as_tuple=True)[0]) for i in range(len(spans))]
    groups.append(("nonimage", (owner == -1).nonzero(as_tuple=True)[0]))
    require(sum(len(keys) for _, keys in groups) == length, "Key groups do not cover every key exactly once")
    return groups


class AttentionRecorder:
    """지정한 층의 attention 입력(hidden, mask, cos/sin)과 o_proj 입력·출력을 forward 한 번에서 기록한다.

    o_proj 출력은 attention 모듈 출력과 같고, 이 기록은 OutputAddition 개입보다 먼저 일어난다 (개입은 모듈 출력 hook).
    """

    def __init__(self, modules, expected_length):
        self.modules, self.expected_length = dict(modules), expected_length
        self.records, self.handles = {}, []
        for index, module in self.modules.items():
            self.handles.append(module.register_forward_pre_hook(self._inputs(index), with_kwargs=True))
            self.handles.append(module.o_proj.register_forward_hook(self._projection(index)))

    def _inputs(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.shape[1] != self.expected_length:
                return None
            require(hidden.shape[0] == 1, "Contribution extraction supports batch size 1 only")
            require(kwargs.get("past_key_values") is None, "Run the extraction forward with use_cache=False")
            require(getattr(module, "sliding_window", None) is None, "Sliding-window attention is not supported")
            cos, sin = kwargs["position_embeddings"]
            self.records[index] = {"hidden": hidden.detach(), "mask": kwargs.get("attention_mask"),
                                   "cos": cos.detach(), "sin": sin.detach()}
            return None
        return hook

    def _projection(self, index):
        def hook(module, args, output):
            record = self.records.get(index)
            if record is not None and "o" not in record and args[0].shape[1] == self.expected_length:
                record["z"], record["o"] = args[0].detach(), output.detach()
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def recompute_attention(module, record, queries, compute_dtype=torch.float32):
    """query 행의 attention 확률 A [1, H, Q, T]와 head별 value [1, H, T, d] (compute_dtype)."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb, repeat_kv

    hidden = record["hidden"]
    length, head_dim = hidden.shape[1], module.head_dim
    with torch.no_grad():
        q = module.q_proj(hidden).view(1, length, -1, head_dim).transpose(1, 2)
        k = module.k_proj(hidden).view(1, length, -1, head_dim).transpose(1, 2)
        v = module.v_proj(hidden).view(1, length, -1, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, record["cos"], record["sin"])
        k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        rows = torch.as_tensor(queries, device=hidden.device)
        scores = q[:, :, rows].to(compute_dtype) @ k.to(compute_dtype).transpose(2, 3) * module.scaling
        mask = record["mask"]
        if mask is None:
            allowed = torch.arange(length, device=hidden.device)[None, :] <= rows[:, None]
            scores = scores.masked_fill(~allowed, float("-inf"))
        elif mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask[:, :, rows, :length], float("-inf"))
        else:
            scores = scores + mask[:, :, rows, :length].to(compute_dtype)
        probs = torch.softmax(scores, dim=-1)
    return probs, v.to(compute_dtype)


def contributions(module, record, queries, groups, compute_dtype=torch.float32):
    """묶음별 c_g [Q, hidden] (o_proj 이후, bias 제외)와 분해하지 않은 참조 출력 (같은 A·v·W_O)."""
    probs, values = recompute_attention(module, record, queries, compute_dtype)
    count = probs.shape[2]
    weight = module.o_proj.weight.detach().to(compute_dtype)
    bias = module.o_proj.bias
    bias = None if bias is None else bias.detach().to(compute_dtype)
    parts = {}
    with torch.no_grad():
        for name, keys in groups:
            keys = keys.to(probs.device)
            z = (probs[..., keys] @ values[:, :, keys, :]).transpose(1, 2).reshape(count, -1)
            parts[name] = z @ weight.T
        z_all = (probs @ values).transpose(1, 2).reshape(count, -1)
        reference = z_all @ weight.T + (bias if bias is not None else 0)
    return {"parts": parts, "bias": bias, "z_all": z_all, "reference": reference,
            "row_sums": probs.sum(dim=-1), "share": {name: probs[..., keys.to(probs.device)].sum(dim=-1).mean().item()
                                                     for name, keys in groups}}


def reconstruct(parts, bias):
    """o = Σ_g c_g + b_O (bias는 한 번만)."""
    total = torch.stack(list(parts.values())).sum(dim=0)
    return total if bias is None else total + bias


class OutputAddition:
    """지정한 층의 attention 출력(o_proj 뒤, residual 덧셈 앞)에서 지정 토큰에 λ·v를 더한다: o'[t] = o[t] + λ v[ℓ, t].

    vectors: {층: [len(positions), hidden] float32}. 덧셈은 float32로 하고 모델 dtype으로 되돌린다.
    길이가 expected_length인 forward(전체 prefill)에서만 작동한다. 생성 단계(길이 1)는 건드리지 않는다.
    """

    def __init__(self, modules, vectors, positions, scale, expected_length):
        self.vectors, self.scale, self.expected_length = vectors, float(scale), expected_length
        self.positions = torch.as_tensor(positions, dtype=torch.long)
        self.handles = [modules[index].register_forward_hook(self._hook(index)) for index in vectors]

    def _hook(self, index):
        def hook(module, args, output):
            out = output[0]
            if out.shape[1] != self.expected_length:
                return None
            require(out.shape[0] == 1, "The intervention supports batch size 1 only")
            rows = self.positions.to(out.device)
            vector = self.vectors[index].to(out.device, torch.float32)
            changed = out.clone()
            changed[:, rows, :] = (out[:, rows, :].to(torch.float32) + self.scale * vector).to(out.dtype)
            return (changed, *output[1:])
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class OutputProbe:
    """개입 전후 attention 출력을 같은 forward 안에서 비교한다 (개입 hook 앞에 prepend, 뒤에 append).

    층마다 개입 전 출력(pre)과 개입 뒤 출력(post)을 모델 dtype 그대로 보관한다 (비트 단위 비교용).
    """

    def __init__(self, modules, expected_length):
        self.expected_length = expected_length
        self.pre, self.post = {}, {}
        self.handles = [module.register_forward_hook(self._store(index, self.pre), prepend=True)
                        for index, module in modules.items()]
        self._modules = modules

    def attach_post(self):
        """개입 hook을 등록한 뒤 호출한다 (append라서 개입 뒤에 실행된다)."""
        self.handles += [module.register_forward_hook(self._store(index, self.post))
                         for index, module in self._modules.items()]

    def _store(self, index, target):
        def hook(module, args, output):
            if output[0].shape[1] == self.expected_length:
                target[index] = output[0].detach().clone()
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
