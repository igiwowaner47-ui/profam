from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src.data.objects import ProteinDocument
from src.models.base import BaseFamilyLitModule


class LlamaLitModule(BaseFamilyLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        num_decay_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        pass_res_pos_in_doc_as_position_ids: bool = True,
        max_sequence_index: int = 1024,
        optimizer: str = "adamw",
        override_optimizer_on_load: bool = False,
        gym_results_save_dir=None,
        # New loss: zero gradients for samples whose mean log-likelihood exceeds a threshold
        gym_subsamples_per_n: int = 5,
    ) -> None:
        """
        From the paper:
        We trained using the AdamW optimizer (Loshchilov and Hutter, 2017),
        with beta1=0.9,beta2=0.95,eps=10-5. We use a cosine learning rate schedule, with warmup
        of 2000 steps, and decay final learning rate down to 10% of the peak learning rate (3e-4-1.5e-4).
        We use a weight decay of 0.1 and gradient clipping of 1.0.
        """
        model = LlamaForCausalLM(config)
        # n.b. attention implementation gets set here (in from_pretrained, _from_config, __init__):
        # https://github.com/huggingface/transformers/blob/1dba608df93ffb10a9c268ef35191adf2424c5ca/src/transformers/modeling_utils.py#L1542
        # c.f. https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
        print(
            "Initialised Llama model, attention implementation: ",
            model.config._attn_implementation,
        )

        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            override_optimizer_on_load=override_optimizer_on_load,
            pass_res_pos_in_doc_as_position_ids=pass_res_pos_in_doc_as_position_ids,
        )
        self.family_flow_projection: Optional[torch.nn.Linear] = None

    def _prepare_family_flow_inputs(
        self,
        aa_seq: Union[str, Sequence[str]],
        wt_name: Optional[Union[str, Sequence[Optional[str]]]],
    ) -> tuple[List[str], List[Optional[str]]]:
        if isinstance(aa_seq, str):
            sequences = [aa_seq]
        else:
            sequences = list(aa_seq)

        if len(sequences) == 0:
            raise ValueError("aa_seq must contain at least one sequence")
        if any(not seq for seq in sequences):
            raise ValueError("aa_seq must not contain empty sequences")

        if wt_name is None or isinstance(wt_name, str):
            wt_names = [wt_name] * len(sequences)
        else:
            wt_names = list(wt_name)
            if len(wt_names) != len(sequences):
                raise ValueError("wt_name and aa_seq must have matching lengths")

        return sequences, wt_names

    def extract_family_flow_features(
        self,
        aa_seq: Union[str, Sequence[str]],
        wt_name: Optional[Union[str, Sequence[Optional[str]]]] = None,
        projection_layer: Optional[torch.nn.Module] = None,
        projection_dim: Optional[int] = None,
        inference_mode: bool = True,
    ) -> Dict[str, Any]:
        """Extract bidirectional ProFam family-flow features.

        This keeps the causal mask untouched and obtains bidirectional context via
        sequence reversal:
        1) forward pass on the original sequence to get ``E_fwd``
        2) forward pass on the reversed sequence to get ``E_rev``
        3) reverse ``E_rev`` over sequence dimension to get ``E_bwd``
        4) concatenate ``[E_fwd, E_bwd]`` on feature dim

        Args:
            aa_seq: Amino-acid sequence(s).
            wt_name: Optional sequence identifier(s) for downstream bookkeeping.
            projection_layer: Optional projection module (e.g. ``nn.Linear``)
                to align ``features`` to another backbone dimension.
            projection_dim: Optional dimension for an internal learnable linear
                projector (used when ``projection_layer`` is not provided).
            inference_mode: If True, disable grad tracking for extraction.
                Set False when the projection should be trainable end-to-end.

        Returns:
            Dict containing ``features``/``Zconcat`` with shape [B, L, 2H] and
            intermediate tensors ``E_fwd``, ``E_rev``, ``E_bwd`` (all [B, L, H]).
            If projection is enabled, ``projected_features``/``Zfam`` is also
            returned.
        """
        sequences, wt_names = self._prepare_family_flow_inputs(aa_seq, wt_name)

        device = self.device
        fwd_docs = [ProteinDocument(sequences=[seq]) for seq in sequences]
        rev_docs = [ProteinDocument(sequences=[seq[::-1]]) for seq in sequences]

        enc_fwd = self.tokenizer.batched_encode(
            fwd_docs,
            document_token="[RAW]",
            padding="longest",
            add_final_sep=False,
            allow_unk=False,
        )
        enc_rev = self.tokenizer.batched_encode(
            rev_docs,
            document_token="[RAW]",
            padding="longest",
            add_final_sep=False,
            allow_unk=False,
        )

        input_ids_fwd = torch.as_tensor(enc_fwd["input_ids"], dtype=torch.long)
        input_ids_rev = torch.as_tensor(enc_rev["input_ids"], dtype=torch.long)
        attention_mask_fwd = torch.as_tensor(
            enc_fwd["attention_mask"], dtype=torch.long
        )
        attention_mask_rev = torch.as_tensor(
            enc_rev["attention_mask"], dtype=torch.long
        )

        # Use the model in its native causal setup and intercept deep hidden states
        # before LM head projection.
        grad_ctx = torch.no_grad() if inference_mode else nullcontext()
        with grad_ctx:
            outputs_fwd = self.model(
                input_ids=input_ids_fwd.to(device),
                attention_mask=attention_mask_fwd.to(device),
                output_hidden_states=True,
                return_dict=True,
            )
            outputs_rev = self.model(
                input_ids=input_ids_rev.to(device),
                attention_mask=attention_mask_rev.to(device),
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_fwd = outputs_fwd.hidden_states[-1]
        hidden_rev = outputs_rev.hidden_states[-1]

        seq_start = self.tokenizer.num_start_tokens
        seq_lengths = [len(seq) for seq in sequences]
        max_len = max(seq_lengths)

        per_seq_e_fwd = []
        per_seq_e_rev = []
        per_seq_e_bwd = []
        for i, seq_len in enumerate(seq_lengths):
            seq_end = seq_start + seq_len
            seq_e_fwd = hidden_fwd[i, seq_start:seq_end, :]
            seq_e_rev = hidden_rev[i, seq_start:seq_end, :]
            per_seq_e_fwd.append(seq_e_fwd)
            per_seq_e_rev.append(seq_e_rev)
            per_seq_e_bwd.append(torch.flip(seq_e_rev, dims=(0,)))

        e_fwd = pad_sequence(per_seq_e_fwd, batch_first=True)
        e_rev = pad_sequence(per_seq_e_rev, batch_first=True)
        e_bwd = pad_sequence(per_seq_e_bwd, batch_first=True)

        seq_mask = (
            torch.arange(max_len, device=device)
            .unsqueeze(0)
            .expand(len(seq_lengths), max_len)
            < torch.tensor(seq_lengths, device=device).unsqueeze(1)
        )
        features = torch.cat([e_fwd, e_bwd], dim=-1)

        out: Dict[str, Any] = {
            "wt_name": wt_names if len(wt_names) > 1 else wt_names[0],
            "seq_lengths": seq_lengths,
            "seq_mask": seq_mask,
            "E_fwd": e_fwd,
            "E_rev": e_rev,
            "E_bwd": e_bwd,
            "features": features,
            "Zconcat": features,
        }
        if projection_layer is not None and projection_dim is not None:
            raise ValueError("Provide either projection_layer or projection_dim, not both")

        if projection_layer is not None:
            projected = projection_layer.to(
                device=device, dtype=features.dtype
            )(features)
            out["projected_features"] = projected
            out["Zfam"] = projected
        elif projection_dim is not None:
            needs_init = (
                self.family_flow_projection is None
                or self.family_flow_projection.in_features != features.shape[-1]
                or self.family_flow_projection.out_features != projection_dim
            )
            if needs_init:
                self.family_flow_projection = torch.nn.Linear(
                    features.shape[-1], projection_dim, bias=True
                )
            self.family_flow_projection = self.family_flow_projection.to(
                device=device, dtype=features.dtype
            )
            projected = self.family_flow_projection(features)
            out["projected_features"] = projected
            out["Zfam"] = projected

        return out
