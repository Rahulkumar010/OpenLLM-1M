import os

from typing import List, Tuple, Union
from web_ui import WebUI
import math

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import RefMaterialOutput, BaseSearch
from qwen_agent.log import logger
from qwen_agent.gui.gradio import gr

POSITIVE_INFINITY = math.inf

@register_tool('no_search')
class NoSearch(BaseSearch):
    def call(self, params: Union[str, dict], docs: List[Union[Record, str, List[str]]] = None, **kwargs) -> list:
        """The basic search algorithm

        Args:
            params: The dict parameters.
            docs: The list of parsed doc, each doc has unique url.

        Returns:
            The list of retrieved chunks from each doc.

        """
        params = self._verify_json_format_args(params)
        # Compatible with the parameter passing of the qwen-agent version <= 0.0.3
        max_ref_token = kwargs.get('max_ref_token', self.max_ref_token)

        # The query is a string that may contain only the original question,
        # or it may be a json string containing the generated keywords and the original question
        if not docs:
            return []
        return self._get_the_front_part(docs, max_ref_token)

    @staticmethod
    def _get_the_front_part(docs: List[Record], max_ref_token: int) -> list:
        all_tokens = 0
        _ref_list = []
        for doc in docs:
            text = []
            for page in doc.raw:
                text.append(page.content)
                all_tokens += page.token
            now_ref_list = RefMaterialOutput(url=doc.url, text=text).to_dict()
            _ref_list.append(now_ref_list)

        logger.info(f'Using tokens: {all_tokens}')
        if all_tokens > max_ref_token:
            raise gr.Error(f"Your document files (around {all_tokens} tokens) exceed the maximum context length ({max_ref_token} tokens).")
        return _ref_list

    def sort_by_scores(self,
                       query: str,
                       docs: List[Record],
                       max_ref_token: int,
                       **kwargs) -> List[Tuple[str, int, float]]:
        raise NotImplementedError

def app_gui():
    # Define the agent
    bot135 = Assistant(llm={
                    'model': 'SmolLM2-135M-Instruct',
                    'model_server': 'http://localhost:8000/v1',
                    'api_key': 'EMPTY',
                    'generate_cfg': {
                        'max_input_tokens': 1000000,
                        'max_retries': 10,
                    }},
                    name='Turbo-135M',
                    description='Turbo natively supports input length of up to 1M tokens. You can upload documents for Q&A (eg., pdf/docx/pptx/txt/html)',
                    rag_cfg={'max_ref_token': 1000000, 'rag_searchers': ['no_search']},
                )
    bot360 = Assistant(llm={
                    'model': 'SmolLM2-360M-Instruct',
                    'model_server': 'http://localhost:8000/v1',
                    'api_key': 'EMPTY',
                    'generate_cfg': {
                        'max_input_tokens': 1000000,
                        'max_retries': 10,
                    }},
                    name='Turbo-360M',
                    description='Turbo natively supports input length of up to 1M tokens. You can upload documents for Q&A (eg., pdf/docx/pptx/txt/html)',
                    rag_cfg={'max_ref_token': 1000000, 'rag_searchers': ['no_search']},
                )
    chatbot_config = {
        'input.placeholder': "Type \"/clear\" to clear the history",
        'verbose': True,
    }

    WebUI([bot135, bot360], chatbot_config=chatbot_config).run(enable_mention=True)

if __name__ == '__main__':
    import patching # patch qwen-agent to accelerate 1M processing
    app_gui()
