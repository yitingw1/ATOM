"""
Debug utilities for checking KV cache content after prefill in DCP mode.

Usage: import and call from attention_mla.py's forward_impl_plugin_mode.
These functions print KV cache values stored on each rank after prefill,
to verify that DCP correctly shards KV cache across ranks.

The debug is controlled by environment variable:
    ATOM_DEBUG_KV_CACHE=1  to enable
"""

import logging
import os
import torch
import torch.distributed as dist

logger = logging.getLogger("atom")

_DEBUG_ENABLED = os.environ.get("ATOM_DEBUG_KV_CACHE", "0") == "1"
_DEBUG_PREFILL_DONE = {}  # layer_num -> bool, track if we've printed for this layer
_DEBUG_DECODE_DONE = {}   # layer_num -> int, count decode steps printed


def debug_kv_cache_after_prefill(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_num: int,
    num_decode_tokens: int,
    num_actual_toks: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """Print KV cache content after prefill write, on all ranks.

    Call this AFTER concat_and_cache_mla / concat_and_cache_mla_rope_fused.
    Only prints for the first layer (layer_num==0) to avoid flooding.

    Args:
        kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        slot_mapping: [num_tokens] - maps each token to a slot in kv_cache,
                      PAD_ID (-1) for non-local tokens in DCP
        k_c_normed: [num_tokens, kv_lora_rank] - the kv_c values before cache write
        k_pe: [num_tokens, 1, qk_rope_head_dim] - the k_pe values before cache write
        layer_num: which layer this is
        num_decode_tokens: number of decode tokens (prefill tokens start after this)
        num_actual_toks: total actual tokens
        kv_lora_rank: dimension of kv_c
        qk_rope_head_dim: dimension of k_pe
    """
    if not _DEBUG_ENABLED:
        return
    if layer_num != 0:
        return
    if layer_num in _DEBUG_PREFILL_DONE:
        return
    _DEBUG_PREFILL_DONE[layer_num] = True

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Only print prefill tokens' slot_mapping
    prefill_slot_mapping = slot_mapping[num_decode_tokens:num_actual_toks]

    # Count local vs padded slots
    pad_id = -1
    local_mask = prefill_slot_mapping != pad_id
    num_local = local_mask.sum().item()
    num_total = prefill_slot_mapping.shape[0]
    local_slots = prefill_slot_mapping[local_mask]

    logger.info(f"\n{'='*80}")
    logger.info(f"[KV_DEBUG rank={rank}] Layer {layer_num} - Prefill KV Cache Write")
    logger.info(f"[KV_DEBUG rank={rank}] slot_mapping: total_tokens={num_total}, "
                f"local_tokens={num_local}, padded_tokens={num_total - num_local}")

    if num_total <= 32:
        logger.info(f"[KV_DEBUG rank={rank}] slot_mapping values: "
                    f"{prefill_slot_mapping.cpu().tolist()}")
    else:
        logger.info(f"[KV_DEBUG rank={rank}] slot_mapping first 16: "
                    f"{prefill_slot_mapping[:16].cpu().tolist()}")
        logger.info(f"[KV_DEBUG rank={rank}] slot_mapping last 16: "
                    f"{prefill_slot_mapping[-16:].cpu().tolist()}")

    if num_local > 0:
        logger.info(f"[KV_DEBUG rank={rank}] local slots (non-PAD): "
                    f"{local_slots.cpu().tolist()[:32]}{'...' if num_local > 32 else ''}")

    # Print the input values (k_c_normed, k_pe) BEFORE cache write for prefill tokens
    prefill_kc = k_c_normed[num_decode_tokens:num_actual_toks]
    prefill_kpe = k_pe[num_decode_tokens:num_actual_toks]

    logger.info(f"[KV_DEBUG rank={rank}] k_c_normed (input to cache): "
                f"shape={tuple(prefill_kc.shape)}, "
                f"dtype={prefill_kc.dtype}")

    # Print first 3 tokens' kv_c values (first 8 elements)
    num_print_tokens = min(3, prefill_kc.shape[0])
    for t in range(num_print_tokens):
        kc_vals = prefill_kc[t, :8].float().cpu().tolist()
        logger.info(f"[KV_DEBUG rank={rank}]   token[{t}] k_c_normed[:8] = "
                    f"[{', '.join(f'{v:.6f}' for v in kc_vals)}]")

    # Now read back from kv_cache at the local slots to verify write
    if num_local > 0 and kv_cache.numel() > 0:
        block_size = kv_cache.shape[1]
        kv_dim = kv_cache.shape[2] if kv_cache.dim() == 3 else kv_cache.shape[-1]

        logger.info(f"[KV_DEBUG rank={rank}] kv_cache shape={tuple(kv_cache.shape)}, "
                    f"dtype={kv_cache.dtype}, block_size={block_size}")

        # Read back first few local slots
        num_check = min(3, num_local)
        for i in range(num_check):
            slot = local_slots[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size

            if kv_cache.dim() == 3:
                cached_val = kv_cache[block_idx, block_offset, :].float()
            else:
                cached_val = kv_cache.view(-1, kv_dim)[slot, :].float()

            kv_c_cached = cached_val[:kv_lora_rank]
            k_pe_cached = cached_val[kv_lora_rank:kv_lora_rank + qk_rope_head_dim]

            logger.info(f"[KV_DEBUG rank={rank}]   slot[{slot}] (block={block_idx}, "
                        f"offset={block_offset}):")
            logger.info(f"[KV_DEBUG rank={rank}]     kv_c_cached[:8] = "
                        f"[{', '.join(f'{v:.6f}' for v in kv_c_cached[:8].cpu().tolist())}]")
            logger.info(f"[KV_DEBUG rank={rank}]     k_pe_cached[:8] = "
                        f"[{', '.join(f'{v:.6f}' for v in k_pe_cached[:8].cpu().tolist())}]")

    logger.info(f"{'='*80}\n")

    # Barrier to ensure all ranks print before continuing
    dist.barrier()


def debug_kv_cache_decode(
    kv_cache: torch.Tensor,
    attn_metadata,
    layer_num: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dcp_world_size: int,
):
    """Print KV cache content used during decode, to verify each rank
    sees the correct shard.

    Call this at the beginning of the decode branch in forward_impl_plugin_mode.
    Only prints for the first layer, first 2 decode steps.

    Args:
        kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        attn_metadata: contains decode metadata with paged_kv_indptr, paged_kv_indices
        layer_num: which layer
        kv_lora_rank: dim of kv_c part
        qk_rope_head_dim: dim of k_pe part
        dcp_world_size: DCP world size
    """
    if not _DEBUG_ENABLED:
        return
    if layer_num != 0:
        return

    if layer_num not in _DEBUG_DECODE_DONE:
        _DEBUG_DECODE_DONE[layer_num] = 0
    _DEBUG_DECODE_DONE[layer_num] += 1

    if _DEBUG_DECODE_DONE[layer_num] > 2:
        return

    rank = dist.get_rank()
    step = _DEBUG_DECODE_DONE[layer_num]

    decode_meta = attn_metadata.plugin_metadata.decode
    paged_kv_indptr = decode_meta.paged_kv_indptr
    paged_kv_indices = decode_meta.paged_kv_indices
    paged_kv_last_page_len = decode_meta.paged_kv_last_page_len
    seq_lens = decode_meta.seq_lens
    dcp_tot_seq_lens = getattr(decode_meta, 'dcp_tot_seq_lens', None)

    logger.info(f"\n{'='*80}")
    logger.info(f"[KV_DEBUG rank={rank}] Layer {layer_num} - Decode Step {step}")
    logger.info(f"[KV_DEBUG rank={rank}] dcp_world_size={dcp_world_size}")
    logger.info(f"[KV_DEBUG rank={rank}] seq_lens (dcp_local): {seq_lens.cpu().tolist()}")
    if dcp_tot_seq_lens is not None:
        logger.info(f"[KV_DEBUG rank={rank}] dcp_tot_seq_lens: {dcp_tot_seq_lens.cpu().tolist()}")
    logger.info(f"[KV_DEBUG rank={rank}] paged_kv_indptr: {paged_kv_indptr.cpu().tolist()}")

    num_indices = paged_kv_indices.shape[0]
    if num_indices <= 32:
        logger.info(f"[KV_DEBUG rank={rank}] paged_kv_indices: "
                    f"{paged_kv_indices.cpu().tolist()}")
    else:
        logger.info(f"[KV_DEBUG rank={rank}] paged_kv_indices (first 16): "
                    f"{paged_kv_indices[:16].cpu().tolist()}")
        logger.info(f"[KV_DEBUG rank={rank}] paged_kv_indices (last 16): "
                    f"{paged_kv_indices[-16:].cpu().tolist()}")

    logger.info(f"[KV_DEBUG rank={rank}] paged_kv_last_page_len: "
                f"{paged_kv_last_page_len.cpu().tolist()}")

    # Read KV cache content from first few blocks
    if kv_cache.numel() > 0:
        block_size = kv_cache.shape[1]
        kv_dim = kv_cache.shape[2] if kv_cache.dim() == 3 else kv_cache.shape[-1]

        # Print content from first 2 blocks referenced by paged_kv_indices
        num_blocks_to_check = min(2, num_indices)
        for bi in range(num_blocks_to_check):
            block_id = paged_kv_indices[bi].item()
            # Print first slot in this block
            if kv_cache.dim() == 3:
                cached_val = kv_cache[block_id, 0, :].float()
            else:
                cached_val = kv_cache.view(-1, kv_dim)[block_id * block_size, :].float()

            kv_c_val = cached_val[:kv_lora_rank]
            k_pe_val = cached_val[kv_lora_rank:kv_lora_rank + qk_rope_head_dim]

            logger.info(f"[KV_DEBUG rank={rank}]   block[{block_id}] slot0:")
            logger.info(f"[KV_DEBUG rank={rank}]     kv_c[:8] = "
                        f"[{', '.join(f'{v:.6f}' for v in kv_c_val[:8].cpu().tolist())}]")
            logger.info(f"[KV_DEBUG rank={rank}]     k_pe[:8] = "
                        f"[{', '.join(f'{v:.6f}' for v in k_pe_val[:8].cpu().tolist())}]")

            # Also check if all values in this block are zero (uninitialized)
            is_all_zero = (cached_val.abs().sum().item() == 0.0)
            logger.info(f"[KV_DEBUG rank={rank}]     all_zero={is_all_zero}, "
                        f"norm={cached_val.norm().item():.6f}")

    logger.info(f"{'='*80}\n")

    dist.barrier()
