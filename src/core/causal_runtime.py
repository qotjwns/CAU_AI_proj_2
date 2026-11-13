"""STEP 5-3 개입 실행 핵심 (GPU, 기준선과 같은 bf16·sdpa).

빠른 경로: 개입이 바꾸는 것은 앞부분(prefix) 뒤 토큰의 계산뿐이다. 앞 토큰은 뒤 토큰을 보지 않으므로(인과 attention)
prefix는 입력당 한 번 계산(KV cache)하고, 뒤쪽(suffix)만 조건별로 묶어서(batch) 다시 계산한다.
- 실험 A: prefix = 마지막 입력 토큰 앞까지, suffix = 마지막 입력 토큰 1개. 층 ℓ 출력 residual(= 층 ℓ+1 입력, ℓ = 36은
  최종 norm 입력)을 donor 값으로 바꾼다.
- 실험 B: prefix = 마지막 이미지의 vision_end 표시까지, suffix = 이미지 뒤 텍스트 토큰(질문 + 템플릿, 26~28개).
  지정한 층 구간에서 지정한 query 행 → 대상 이미지 토큰 attention logit을 바꾼다 (차단 −∞에 해당하는 dtype 최솟값,
  강화 +log 4를 mask dtype으로 표현한 값).
무개입 기준선도 같은 경로·같은 batch에서 계산한다. 생성은 HF generate와 같은 greedy 규칙(반복 패널티 포함)으로
이어 간다. 참조 경로(연습 실행 전용)는 기준 추출과 같은 model.generate 전체 prefill에 같은 개입을 넣는다.
"""
import math
import time

import torch

from src.core.features import map_image_tokens
from src.core.feature_validation import require
from src.core.inference import prepare_inputs
from src.core.prompts import parse_prediction


LOG4 = math.log(4.0)
SPAN_KEYS = ("start", "end", "token_count")
UNSUPPORTED_GENERATION = ("no_repeat_ngram_size", "bad_words_ids", "forced_bos_token_id", "forced_eos_token_id",
                          "suppress_tokens", "begin_suppress_tokens", "sequence_bias", "exponential_decay_length_penalty",
                          "encoder_repetition_penalty", "guidance_scale", "min_new_tokens")


class Hooks:
    """디코더 층·attention·최종 norm에 한 번 등록하고, 상태로 켜고 끈다. 꺼져 있으면 아무것도 바꾸지 않는다."""

    def __init__(self, model):
        self.decoder = model.get_decoder()
        self.layers = self.decoder.layers
        self.n_layers = len(self.layers)
        self.reset()
        self.handles = []
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._residual_hook(index), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._attention_hook(index),
                                                                           with_kwargs=True))
        self.handles.append(self.decoder.norm.register_forward_pre_hook(self._residual_hook(self.n_layers),
                                                                        with_kwargs=True))
        self.handles.append(self.decoder.norm.register_forward_hook(self._final_hook))

    def reset(self):
        self.active = False
        self.only_length = None       # 참조 경로: 이 길이(전체 prefill)일 때만 개입·포착
        self.patch_layer = None       # [B] long, -1 = 이식 없음
        self.donor = None             # [B, L + 1, H]
        self.position = -1            # 이식·포착 위치 (빠른 경로는 suffix 마지막)
        self.capture = None           # dict: residual index → [B, H]
        self.final = None             # 최종 norm 출력 [B, H]
        self.attention = None         # {"layers": [B, L] bool, "select": [B,1,S,K] bool, "value": [B,1,S,K]}
        self.reference_attention = None   # {"layers": set, "select": [1,1,T,T] bool, "value": [1,1,T,T]}

    def _applies(self, hidden):
        return self.active and (self.only_length is None or hidden.shape[1] == self.only_length)

    def _residual_hook(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if not self._applies(hidden):
                return None
            if self.capture is not None:
                self.capture[index] = hidden[:, self.position, :].detach().clone()
            if self.patch_layer is None:
                return None
            rows = (self.patch_layer == index).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return None
            hidden = hidden.clone()
            hidden[rows, self.position, :] = self.donor[rows, index].to(hidden.dtype)
            if args:
                return (hidden, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": hidden}
        return hook

    def _attention_hook(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if not self._applies(hidden):
                return None
            if self.attention is not None:
                rows = self.attention["layers"][:, index]
                if not bool(rows.any()):
                    return None
                mask = kwargs["attention_mask"]
                require(mask is not None and mask.ndim == 4, "The suffix pass needs an explicit 4D attention mask")
                select = self.attention["select"] & rows[:, None, None, None]
                return args, {**kwargs, "attention_mask": torch.where(select, self.attention["value"], mask)}
            if self.reference_attention is not None and index in self.reference_attention["layers"]:
                require(kwargs.get("attention_mask") is None, "Reference prefill expected the causal (no mask) path")
                return args, {**kwargs, "attention_mask": self.reference_attention["mask"]}
            return None
        return hook

    def _final_hook(self, module, args, output):
        if self._applies(args[0]):
            self.final = output[:, self.position, :].detach().clone()


def causal_mask(query_positions, key_length, dtype, device):
    """[1, 1, Q, K] 가산 mask: query 절대 위치 이하 key는 0, 이후는 dtype 최솟값."""
    keys = torch.arange(key_length, device=device)
    allowed = keys[None, :] <= torch.as_tensor(query_positions, device=device)[:, None]
    return torch.where(allowed, torch.tensor(0, dtype=dtype, device=device),
                       torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device))[None, None]


class Runner:
    def __init__(self, model, processor, answer_ids, variants, max_new_tokens=8):
        from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

        self.model, self.processor = model, processor
        self.tokenizer = processor.tokenizer
        self.hooks = Hooks(model)
        self.n_layers = self.hooks.n_layers
        self.dtype = next(self.hooks.decoder.parameters()).dtype
        self.device = model.device
        self.max_new_tokens = max_new_tokens
        config = model.generation_config
        for name in UNSUPPORTED_GENERATION:
            value = getattr(config, name, None)
            require(value in (None, 0, 1.0, False) or (name == "encoder_repetition_penalty" and value == 1.0),
                    f"Generation option {name}={value!r} is not replicated by the fast path")
        require(not getattr(config, "min_length", 0), "min_length is not replicated by the fast path")
        penalty = getattr(config, "repetition_penalty", None)
        self.repetition_penalty = penalty
        self.processors = LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=penalty)] if penalty not in (None, 1.0) else [])
        eos = config.eos_token_id
        self.eos = torch.tensor(eos if isinstance(eos, (list, tuple)) else [eos], device=self.device)
        self.pad = config.pad_token_id if config.pad_token_id is not None else int(self.eos[0])
        self.yes_id, self.no_id = answer_ids
        self.variants = variants
        head = model.get_output_embeddings()
        require(getattr(head, "bias", None) is None, "fp32 margin assumes an output head without bias")
        rows = sorted({self.yes_id, self.no_id, *variants["Yes"], *variants["No"]})
        self.head_rows = head.weight.detach()[rows].double().cpu()
        self.row_of = {token: i for i, token in enumerate(rows)}
        self.boost_value = torch.tensor(LOG4, dtype=self.dtype)
        self.block_value = torch.tensor(torch.finfo(self.dtype).min, dtype=self.dtype)

    # ------------------------------------------------------------ 입력

    def prepare(self, case, baseline=None):
        inputs = prepare_inputs(case, self.model, self.processor)
        spans, query_index = map_image_tokens(inputs, self.model.config, self.processor.image_processor.merge_size,
                                              case["image_ids"])
        length = inputs["input_ids"].shape[1]
        require(query_index == length - 1, "The input has padding; the fast path expects none")
        if baseline is not None:
            require([{k: s[k] for k in SPAN_KEYS} for s in spans]
                    == [{k: s[k] for k in SPAN_KEYS} for s in baseline["image_spans"]]
                    and query_index == baseline["query_index"],
                    f"{case['case_id']}: image token spans differ from the baseline run")
        positions, _ = self.model.model.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
        return {"case": case, "inputs": inputs, "spans": spans, "length": length, "positions": positions}

    def prefix(self, prepared, prefix_length):
        inputs = prepared["inputs"]
        require(0 < prefix_length < prepared["length"], "Invalid prefix length")
        require(prepared["spans"][-1]["end"] < prefix_length, "Every image token must be inside the prefix")
        self.hooks.reset()
        with torch.inference_mode():
            output = self.model(input_ids=inputs["input_ids"][:, :prefix_length], pixel_values=inputs["pixel_values"],
                                image_grid_thw=inputs["image_grid_thw"],
                                position_ids=prepared["positions"][:, :, :prefix_length], use_cache=True,
                                logits_to_keep=1)
        cache = [(layer.keys, layer.values) for layer in output.past_key_values.layers]
        require(len(cache) == self.n_layers and all(k.shape[-2] == prefix_length for k, _ in cache),
                "Unexpected prefix cache shape")
        return {"length": prefix_length, "cache": cache}

    def expanded_cache(self, prefix, batch):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        for index, (keys, values) in enumerate(prefix["cache"]):
            cache.update(keys.expand(batch, -1, -1, -1), values.expand(batch, -1, -1, -1), index)
        return cache

    # ------------------------------------------------------------ suffix와 생성

    def run_suffix(self, prepared, prefix, batch, patch=None, attention=None, capture=False):
        """suffix를 batch 행으로 계산하고 greedy 생성까지 한다. 행마다 답·margin을 돌려준다.

        patch: {"layer": [B] long (-1 = 없음), "donor": [B, L+1, H]} (실험 A)
        attention: {"layers": [B, L] bool, "select": [B, S, T] bool, "value": [B] 값} (실험 B)
        """
        inputs, start = prepared["inputs"], prefix["length"]
        length = prepared["length"]
        suffix = length - start
        input_ids = inputs["input_ids"]
        positions = prepared["positions"][:, :, start:].expand(3, batch, suffix)
        cache = self.expanded_cache(prefix, batch)
        hooks = self.hooks
        hooks.reset()
        mask = None
        if attention is not None:
            base = causal_mask(list(range(start, length)), length, self.dtype, self.device)
            mask = base.expand(batch, 1, suffix, length).contiguous()
            value = attention["value"].to(self.device, self.dtype)[:, None, None, None].expand(batch, 1, suffix, length)
            hooks.attention = {"layers": attention["layers"].to(self.device),
                               "select": attention["select"].to(self.device)[:, None], "value": value}
        if patch is not None:
            hooks.patch_layer = patch["layer"].to(self.device)
            hooks.donor = patch["donor"].to(self.device)
        hooks.capture = {} if capture else None
        hooks.active = True
        try:
            with torch.inference_mode():
                output = self.model(input_ids=input_ids[:, start:].expand(batch, suffix), position_ids=positions,
                                    past_key_values=cache, attention_mask=mask, use_cache=True, logits_to_keep=1)
        finally:
            hooks.active = False
        final, captured = hooks.final, hooks.capture
        hooks.reset()
        logits = output.logits[:, -1, :].to(dtype=torch.float32)
        tokens = self.greedy(input_ids.expand(batch, length), logits, output.past_key_values,
                             prepared["positions"][:, :, -1:])
        rows = self.summaries(logits, final, tokens)
        residuals = None
        if capture:
            residuals = torch.stack([captured[i] for i in range(self.n_layers + 1)], dim=1).cpu()
        return rows, residuals

    def greedy(self, sequences, logits, cache, last_position):
        """HF greedy와 같은 규칙: float32 점수 → logits processor(반복 패널티) → argmax, EOS 뒤에는 pad."""
        batch = sequences.shape[0]
        sequences = sequences.clone()
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        generated = []
        for step in range(self.max_new_tokens):
            if step > 0:
                position = (last_position + step).expand(3, batch, 1)
                with torch.inference_mode():
                    output = self.model(input_ids=sequences[:, -1:], position_ids=position, past_key_values=cache,
                                        use_cache=True, logits_to_keep=1)
                logits = output.logits[:, -1, :].to(dtype=torch.float32)
            scores = self.processors(sequences, logits.clone())
            following = scores.argmax(dim=-1)
            following = torch.where(finished, torch.full_like(following, self.pad), following)
            generated.append(following)
            sequences = torch.cat([sequences, following[:, None]], dim=1)
            finished |= torch.isin(following, self.eos)
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1).cpu()

    def summaries(self, logits, final, tokens):
        logits = logits.cpu()
        values = self.head_rows @ final.cpu().double().T            # [rows, B], float64
        yes_rows = [self.row_of[t] for t in self.variants["Yes"]]
        no_rows = [self.row_of[t] for t in self.variants["No"]]
        rows = []
        for b in range(logits.shape[0]):
            raw = self.tokenizer.decode(tokens[b].tolist(), skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            yes, no = logits[b, self.yes_id].item(), logits[b, self.no_id].item()
            variant = (logits[b, self.variants["Yes"]].logsumexp(0) - logits[b, self.variants["No"]].logsumexp(0)).item()
            column = values[:, b]
            rows.append({
                "answer": parse_prediction(raw), "raw": raw, "first_token": int(tokens[b, 0]),
                "margin_bf16": yes - no, "margin_bf16_variant": variant,
                "margin_fp32": (column[self.row_of[self.yes_id]] - column[self.row_of[self.no_id]]).item(),
                "margin": (column[yes_rows].logsumexp(0) - column[no_rows].logsumexp(0)).item(),
            })
        return rows

    # ------------------------------------------------------------ 참조 경로 (연습 실행 전용)

    def reference(self, prepared, patch_layer=None, donor=None, attention=None):
        """기준 추출과 같은 model.generate(전체 prefill)에 같은 개입을 넣는다. 한 조건씩 (batch 1)."""
        inputs, length = prepared["inputs"], prepared["length"]
        hooks = self.hooks
        hooks.reset()
        hooks.only_length = length
        if patch_layer is not None:
            hooks.patch_layer = torch.tensor([patch_layer], device=self.device)
            hooks.donor = donor[None].to(self.device)
        if attention is not None:
            mask = reference_attention_mask(length, attention["queries"], attention["targets"], attention["value"],
                                            self.dtype, self.device)
            hooks.reference_attention = {"layers": set(attention["layers"]), "mask": mask}
        hooks.active = True
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                                                return_dict_in_generate=True, output_logits=True)
        finally:
            hooks.active = False
        final = hooks.final
        hooks.reset()
        tokens = generated.sequences[:, length:].cpu()
        logits = generated.logits[0].to(dtype=torch.float32)
        return self.summaries(logits, final, tokens)[0]


def reference_attention_mask(length, queries, targets, value, dtype, device):
    """참조 경로용 전체 mask [1,1,T,T] (빠른 경로와 별도로 만든다): 인과 mask에 (query 행 × 대상 열) 값을 넣는다."""
    minimum = torch.finfo(dtype).min
    mask = torch.full((length, length), minimum, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)                      # 대각선 아래·대각선은 0
    for query in queries:
        for start, end in targets:
            mask[query, start:end] = value
    return mask[None, None]


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
