from typing import List, Optional

from vllm.config import VLLMConfig
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class CtcProposer:
    def __init__(self, vllm_config: VLLMConfig):
        self.vllm_config = vllm_config
        self.spec_config = vllm_config.speculative_config
        # CTC text and pointer state per request
        self.ctc_texts: dict[int, str] = {}  # req_id -> ctc_text
        self.pointers: dict[int, int] = {}   # req_id -> current_position

    def set_ctc_text(self, req_id: int, ctc_text: str):
        """Set the CTC text for a request and initialize pointer."""
        self.ctc_texts[req_id] = ctc_text
        self.pointers[req_id] = 0

    def propose(self, req_id: int, prompt_tokens: List[int], metadata: SpecDecodeMetadata) -> List[List[int]]:
        """Propose draft tokens using CTC pointer method."""
        if req_id not in self.ctc_texts:
            return []

        ctc_text = self.ctc_texts[req_id]
        pointer = self.pointers[req_id]

        if pointer >= len(ctc_text):
            # No more CTC text to propose
            return []

        # Get remaining CTC text from pointer
        remaining_text = ctc_text[pointer:]
        max_concat = self.spec_config.ctc_max_concat_length or len(remaining_text)
        concat_text = remaining_text[:max_concat]

        # For now, return empty list as placeholder
        # In full implementation, this would tokenize and return the concat_text
        # But actual proposal logic would be handled in the verification step
        draft_token_ids: List[List[int]] = []

        return draft_token_ids
    

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    def update_pointer(self, req_id: int, accepted_length: int):
        """Update pointer based on accepted length."""
        if req_id in self.pointers:
            self.pointers[req_id] += accepted_length

    def get_current_pointer(self, req_id: int) -> int:
        """Get current pointer position for a request."""
        return self.pointers.get(req_id, 0)

    def clear_request(self, req_id: int):
        """Clear CTC data for a completed request."""
        self.ctc_texts.pop(req_id, None)
        self.pointers.pop(req_id, None)
