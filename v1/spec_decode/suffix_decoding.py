# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch
# kyleryu
from vllm.logger import init_logger
logger = init_logger(__name__)


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        self.num_speculative_tokens = config.num_speculative_tokens 
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len
        # kyleryu - 获取tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer,
            trust_remote_code=vllm_config.model_config.trust_remote_code
        )
        self.tokenizer = tokenizer
        
        # 获取 CTC 标记的 token IDs
        self.ctc_start_ids = tokenizer.encode("<CTC>", add_special_tokens=False)
        self.ctc_end_ids = tokenizer.encode("</CTC>", add_special_tokens=False)
        
        
        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        # kyleryu  
        # 下面这部分不需要了
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    # kyleryu
    # 提取ctc文本
    def _ectract_ctc_text(self, prompt_token_ids: list[int]):
        prompt_list = prompt_token_ids.tolist()#转换成list

        start_index = None
        for i in range(len(prompt_list)-len(self.ctc_start_ids) + 1):
            if prompt_list[i:i+len(self.ctc_start_ids)] == self.ctc_start_ids:
                start_index = i + len(self.ctc_start_ids)
                break
        if start_index is None:
            return None
        
        end_index = None
        for i in range(start_index, len(prompt_list)-len(self.ctc_end_ids) + 1):
            if prompt_list[i:i+len(self.ctc_end_ids)] == self.ctc_end_ids:
                end_index = i
                break
        if end_index is None:
            return None
        ctc_token_ids = prompt_list[start_index:end_index]
        return ctc_token_ids


    # kyleryu
    # 逐请求生成draft tokens
    # 输入参数:input_batch:这一批并发请求的状态，包含每个请求的 token 序列、prompt 长度、req_id 等状态
    # sampled_token_ids:上一轮/本轮真实模型已经采样并提交的tokens，即已被确认的tokens
    # 输出参数:draft_token_ids:对于batch里的请求，返回一个list[int]作为草稿
    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        # kyleryu
        # 缓存ctc文本
        if not hasattr(self,'ctc_cache'):
            self.ctc_cache = {} # ctc文本  {req_id: ctc_token_ids} 字典
            self.ctc_pointer = {}# ctc指针 {req_id: current_position}


        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding. 跳过不支持的
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length. 超长
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            # if req_id not in self.suffix_cache.active_requests:
                # kyleryu- 删除suffix
                # if req_id in self.suffix_cache.cached_requests:
                #     # Reset the suffix cache for this request.
                #     self.suffix_cache.evict_cached_response(req_id)
            if req_id not in self.ctc_cache: #首次处理该请求，提取ctc
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # kyleryu
                print(f'num_prompt_tokens={num_prompt_tokens}')
                print("==== DEBUG PROMPT TOKENS BEGIN ====")
                print(prompt_token_ids)  
                print("==== DEBUG PROMPT TOKENS END ====")

                ctc_token_ids = self._ectract_ctc_text(prompt_token_ids)
                if ctc_token_ids is not None:
                    self.ctc_cache[req_id] = ctc_token_ids
                    self.ctc_pointer[req_id] = 0 # 初始化指针
                    ctc_text = self.tokenizer.decode(ctc_token_ids,skip_special_tokens=True)
                   
                else:
                    print('No CTC text found.')
                    self.ctc_cache[req_id] = []
                    self.ctc_pointer[req_id] = 0

                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)
            # kyleryu
            logger.info("logkyleryu_n_reqs=%d lens=%s", len(sampled_token_ids),
                        [len(x) for x in sampled_token_ids][:8])
            logger.info("logkyleryu_draft_lens=%s", [len(x) for x in draft_token_ids][:8])



        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)
        print(f'kyleryulog_SuffixDecoding proposer drafted tokens for {len(draft_token_ids)} requests.')
        # 打印sampled tokens
        print(f'kyleryulog_sampled_token_ids={sampled_token_ids}')
        # 打印草稿tokens
        print(f'kyleryulog_draft_token_ids={draft_token_ids}')

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
