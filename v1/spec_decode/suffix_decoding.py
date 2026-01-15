# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch
# kyleryu
from vllm.logger import init_logger
logger = init_logger(__name__)


class SuffixDecodingProposer:
    """
    CTC-based speculative decoding proposer. 
    Uses CTC recognition results as draft tokens for speculative decoding.
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        self.num_speculative_tokens = config.num_speculative_tokens 
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
        
        # kyleryu - 添加计数器
        self.step_count = 0
        self.window = 8

    # kyleryu - 提取ctc文本
    def _extract_ctc_text(self, prompt_token_ids):
        """从 prompt 中提取 <CTC>... </CTC> 之间的 token IDs"""
        prompt_list = prompt_token_ids.tolist()  # 转换成list

        # 查找 <CTC> 标签
        start_index = None
        for i in range(len(prompt_list) - len(self.ctc_start_ids) + 1):
            if prompt_list[i:i+len(self.ctc_start_ids)] == self.ctc_start_ids:
                start_index = i + len(self.ctc_start_ids)
                break
        if start_index is None:
            return None
        
        # 查找 </CTC> 标签
        end_index = None
        for i in range(start_index, len(prompt_list) - len(self.ctc_end_ids) + 1):
            if prompt_list[i:i+len(self.ctc_end_ids)] == self.ctc_end_ids:
                end_index = i
                break
        if end_index is None:
            return None
        
        ctc_token_ids = prompt_list[start_index:end_index]
        return ctc_token_ids
    def _update_ctc_pointer(self, req_id,last_draft,sampled_ids) -> None:
        """根据已接受的 token 数量更新 CTC 指针
        Args:
            req_id (int): The request ID.
            sampled_ids (list[int]): 本轮被验证通过的
            last_draft (list[int]): 上一轮提议的draft tokens # 目前存疑❓️
        """
        if req_id not in self.ctc_cache or req_id not in self.ctc_pointer:
            return
        ctc_tokens = self.ctc_cache[req_id] # ctc的token列表
        old_pointer = self.ctc_pointer[req_id] # 旧指针位置
         # ============ 阶段 1: 计算匹配长度 ============
        # sampled_ids 的结构: [draft_0, draft_1, ..., draft_k, correction]
        # 我们需要找出有多少个 draft token 被验证通过
        match_len = 0
        if len(last_draft) > 0 and len(sampled_ids) > 0:
            # 逐个对比 sampled_ids 和 last_draft    [draft_0, draft_1, ..., draft_k]   [draft_0, draft_1, ..., draft_k, draft_k+1]
            # 注意： sampled_ids 的结构：[draft_0, draft_1, ..., draft_k, correction] 最后可能含有修正token
            max_check = min(len(sampled_ids), len(last_draft))  # 
            for i in range(max_check):
                if sampled_ids[i] == last_draft[i]:
                    match_len += 1
                else:
                    break # 遇到第一个不匹配就停止
        print(f' [MATCH CHECK]')
        print(f' Last draft:{last_draft[:10]}{"..." if len(last_draft)>10 else ""}')
        print(f' Sampled IDs:{sampled_ids[:10]}{"..." if len(sampled_ids)>10 else ""}')
        print(f' Match Length:{match_len}/{len(last_draft)}')

        # 阶段1 ：基础指针前进
        self.ctc_pointer[req_id] += match_len

        print(f' [PHASE 1:MATCH UPDATE]')
        print(f' Pointer moved: {old_pointer} -> {self.ctc_pointer[req_id]} (by match length+{match_len})')

        # 阶段2 ：Resync 锚点搜索
        # 获取修正token ，即最后一个sampled token
        if len(sampled_ids) > match_len:
            next_correct_token  = sampled_ids[match_len] # 修正token

            print(f'[CORRECTION TOKEN]')
            try:
                correction_text = self.tokenizer.decode([next_correct_token])
                print(f'  Tokrn ID: {next_correct_token}')
                print(f'  Text: {correction_text}')
            except:
                print(f'  Tokrn ID: {next_correct_token}')
            
            # 在ctc后续部分搜索修正token 【滑窗搜索部分】

            search_start = self.ctc_pointer[req_id]
            found_offset = -1

            if search_start < len(ctc_tokens):
                search_end = min(search_start + self.window, len(ctc_tokens))
                window_tokens = ctc_tokens[search_start:search_end]

                print(f"  [RESYNC SEARCH]")
                print(f"    Search window: CTC[{search_start}:{search_end}]")
                print(f"    Window tokens: {window_tokens}")

                # 搜索锚点
                for k in range(len(window_tokens)):
                    if window_tokens[k] == next_correct_token:
                        found_offset = k
                        break
                if found_offset != -1 :
                    # 找到锚点，更新指针
                    skip_step = found_offset + 1
                    old_ptr_phase2 = self.ctc_pointer[req_id]
                    self.ctc_pointer[req_id] += skip_step


                    print(f"    [ANCHOR FOUND]")
                    print(f"      Offset: {found_offset}")
                    print(f"      Skip tokens: {skip_step}")
                    print(f"      Pointer moved: {old_ptr_phase2} -> {self.ctc_pointer[req_id]}")
                else:
                    # 未找到锚点: 保持指针不变 (CTC 可能有删除错误)
                    print(f"    [ANCHOR NOT FOUND]")
                    print(f"      Pointer unchanged: {self.ctc_pointer[req_id]}")
                    print(f"      Reason: Possible deletion error in CTC")
            else:
                print(f"  [RESYNC SKIPPED]")
                print(f"    Reason: CTC exhausted (pointer={search_start}, total={len(ctc_tokens)})")
        
        # ============ 最终状态 ============
        total_move = self.ctc_pointer[req_id] - old_pointer
        print(f"  [FINAL POINTER UPDATE]")
        print(f"    Total movement: {old_pointer} -> {self.ctc_pointer[req_id]} (+{total_move})")
        print(f"    Remaining CTC: {len(ctc_tokens) - self.ctc_pointer[req_id]}/{len(ctc_tokens)}")



    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request using CTC. 
        """
        draft_token_ids:  list[list[int]] = []
        
        # kyleryu - 初始化缓存
        if not hasattr(self, 'ctc_cache'):
            self.ctc_cache = {}      # {req_id: ctc_token_ids}
            self.ctc_pointer = {}    # {req_id: current_position}
            self.last_draft_cache = {} # {req_id: last_draft_tokens}
        
        # ============ 打印本轮概览 ============
        self.step_count += 1
        print("\n" + "="*100)
        print(f"[STEP {self.step_count}] Batch Processing - {len(sampled_token_ids)} requests")
        print("="*100)

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills. 
                draft_token_ids.append([])
                continue

            # Skip unsupported requests
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that reached max length
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]

            # ============ 打印请求基本信息 ============
            print(f"\n{'─'*100}")
            print(f"[Request {i}] ID: {req_id}")
            print(f"  Current length: {num_tokens} tokens")

            # ============ 首次处理：提取 CTC ============
            # 0115======== 添加验证首次生成token =========
            if req_id not in self.ctc_cache:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, : num_prompt_tokens]
                
                print(f"  [FIRST TIME] Extracting CTC from prompt ({num_prompt_tokens} tokens)")

                ctc_token_ids = self._extract_ctc_text(prompt_token_ids)
                if ctc_token_ids is not None:
                    self.ctc_cache[req_id] = ctc_token_ids
                    self.ctc_pointer[req_id] = 0
                    self.last_draft_cache[req_id] = []
                    ctc_text = self.tokenizer.decode(ctc_token_ids, skip_special_tokens=True)
                    
                    print(f"  [CTC] Extracted {len(ctc_token_ids)} tokens")
                    print(f"  [CTC] Token IDs: {ctc_token_ids[: 20]}{'...' if len(ctc_token_ids) > 20 else ''}")
                    print(f"  [CTC] Text: '{ctc_text}'")
            # 0115============ 验证首token ============
                if len(sampled_ids) > 0:
                    print(f'\n  [VERIFY FIRST TOKEN]')
                    print(f'  First sampled token ID: {sampled_ids}')

                    # 检查已生成的token是否匹配CTC开头
                    match_count = 0
                    for j,token in enumerate(sampled_ids):
                        if j < len(ctc_token_ids) and token == ctc_token_ids[j]:
                            match_count += 1
                        else:
                            break
                    if match_count > 0:
                        self.ctc_pointer[req_id] = match_count
                        print(f" ✓ Matched {match_count}/{len(sampled_ids)} tokens with CTC")
                        print(f"    Pointer initialized to: {match_count}/{len(ctc_token_ids)}")
                        
                        try:
                            matched_text = self.tokenizer.decode(ctc_token_ids[: match_count], skip_special_tokens=True)
                            print(f"    Matched Text: '{matched_text}'")
                        except Exception as e:
                            pass
                    else:
                        print(f" ✗ No match found")
                        print(f' Pointer unchanged: 0/0')
                    
                else:
                    print(f"  [CTC] Not found in prompt")
                    self.ctc_cache[req_id] = []
                    self.ctc_pointer[req_id] = 0
                    self.last_draft_cache[req_id] = []

            # # 0115============ 非首次: 更新指针 (CPV 风格) ============
            else:
                if len(sampled_ids) > 0 and req_id in self.last_draft_cache:
                    last_draft = self.last_draft_cache[req_id]
                    
                    print(f"\n  [POINTER UPDATE - CPV STYLE]")
                    self._update_ctc_pointer(req_id, last_draft, sampled_ids)
                else:
                    print(f"\n  [NO POINTER UPDATE]")
                    print(f"    Reason: No sampled tokens or no last draft")




            # # 0114============ 更新指针(根据上一轮接受的token数) ============
            # else:
            #     # sampled_ids 是上一轮被验证通过的tokenid
            #     num_accepted = len(sampled_ids)
            #     if num_accepted > 0 :
            #         old_pointer = self.ctc_pointer[req_id]
            #         self.ctc_pointer[req_id] += num_accepted # ❌️，这里不应该用这个

            #         print(f'[UPDATED POINTER]')
            #         print(f'  Accepted: {num_accepted}tokens from last step')
            #         print(f'  Pointer moved : {old_pointer} -> {self.ctc_pointer[req_id]}')
                
            #         try:
            #             accepted_text = self.tokenizer.decode(sampled_ids, skip_special_tokens=True)
            #             print(f"    Text: '{accepted_text}'")
            #         except Exception as e:
            #             print(f"    Text: [decode error:  {e}]")
            #     else:
            #         print(f'[NO UPDATE]')
            #         print(f'  No tokens accepted from last step; pointer remains at {self.ctc_pointer[req_id]}')


            # ============ 打印本轮接受的 tokens ============
            print(f"\n  [ACCEPTED THIS STEP]")
            print(f"    Token IDs: {sampled_ids}")
            print(f"    Count: {len(sampled_ids)}")
            try:
                accepted_text = self.tokenizer.decode(sampled_ids, skip_special_tokens=True)
                print(f"    Text: '{accepted_text}'")
            except Exception as e:
                print(f"    Text: [decode error:  {e}]")

            # ============ 从 CTC 中获取 draft tokens ============
            if req_id in self.ctc_cache and len(self.ctc_cache[req_id]) > 0:
                pointer = self.ctc_pointer[req_id]
                ctc_tokens = self.ctc_cache[req_id]

                # 计算可以提议的 draft 数量
                remaining = len(ctc_tokens) - pointer
                draft_count = min(
                    self.num_speculative_tokens,
                    remaining,
                    self.max_model_len - num_tokens - 1
                )
                
                if draft_count > 0:
                    draft = ctc_tokens[pointer:pointer + draft_count]
                    
                    print(f"\n  [PROPOSED DRAFT]")
                    print(f"    Token IDs: {draft}")
                    print(f"    Count: {len(draft)}")
                    print(f"    CTC pointer: {pointer}/{len(ctc_tokens)}")
                    try:
                        draft_text = self.tokenizer.decode(draft, skip_special_tokens=True)
                        print(f"    Text: '{draft_text}'")
                    except Exception as e:
                        print(f"    Text: [decode error: {e}]")
                    
                    draft_token_ids.append(draft)
                    self.last_draft_cache[req_id] = draft # 保存本轮draft
                    
                    # TODO: 指针更新应该在验证后进行
                    # self.ctc_pointer[req_id] += accepted_count
                else:
                    print(f"\n  [PROPOSED DRAFT]")
                    print(f"    Token IDs:  []")
                    print(f"    Count: 0")
                    print(f"    Reason: No remaining CTC tokens")
                    draft_token_ids.append([])
                    self.last_draft_cache[req_id] = []
            else:
                print(f"\n  [PROPOSED DRAFT]")
                print(f"    Token IDs:  []")
                print(f"    Count: 0")
                print(f"    Reason:  No CTC cache available")
                draft_token_ids.append([])
                if req_id in self.last_draft_cache:
                    self.last_draft_cache[req_id] = []

        # ============ 打印本轮总结 ============
        print(f"\n{'='*100}")
        print(f"[STEP {self.step_count} SUMMARY]")
        print(f"  Total requests processed:  {len(draft_token_ids)}")
        print(f"  Total accepted tokens: {sum(len(x) for x in sampled_token_ids)}")
        print(f"  Total proposed draft tokens: {sum(len(x) for x in draft_token_ids)}")
        
        # 打印每个请求的详细统计
        print(f"\n  Per-request breakdown:")
        for idx, (sampled, draft) in enumerate(zip(sampled_token_ids, draft_token_ids)):
            if sampled or draft: 
                print(f"    Req[{idx}]:  Accepted={len(sampled)}, Proposed={len(draft)}")
        
        print("="*100 + "\n")

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load for CTC-based proposer
        pass
