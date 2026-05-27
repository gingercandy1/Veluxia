import asyncio
import os
from pathlib import Path

from src.backend.core.model_base import BaseTextGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.backend.core.text.index_memory import ConversationMemory
from src.shared.settings import PROJECT_ROOT


class LlamaGenerator(BaseTextGenerator):
    dynamic = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llama_local  = Path(PROJECT_ROOT) / "models" / "text" / self.model_name
        if isinstance(self.model_filename, str):
            self.llama_path = self.llama_local / self.model_filename
        else:
            print("Not found .gguf file name!")

    def find_gguf(self):
        for i in os.listdir(self.llama_local):
            if os.path.splitext(i)[1] == ".gguf":
                llama_path = self.llama_local / i
                return llama_path
        return None

    def _summarize(self, text: str) -> str:
        prompt = f"Summarize the following conversation concisely:\n\n{text}"
        output = self.pipe.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stream=False,
        )
        return output["choices"][0]["message"]["content"].strip()

    def _check_model_file(self):
        if not self.llama_local.exists() or not self.llama_path.exists():
            print("⏬ 正在下载 权重...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=str(self.model_id),
                local_dir=str(self.llama_local),
                filename=str(self.model_filename),
                token=huggingface_token
            )
            print("✅ NVFP4 transformer 下载完成")

    def _load_model(self):
        if self.pipe is not None:
            return

        # n_gpu_layers 先尝试 -1（全部 offload 到 GPU），如果 OOM 就改成 20~35
        # n_ctx 上下文长度，建议从 4096 开始，8GB VRAM 下不要超过 8192~16384
        # n_batch 批处理大小，可适当调大加速
        # verbose 打开日志，方便看是否 offload 到 GPU
        from llama_cpp import Llama, llama_cpp
        print("🔧 正在加载模型（首次较慢）...")
        self.pipe = Llama(
            model_path=str(self.llama_path),
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=512,
            verbose=False,
        )
        print(llama_cpp.llama_backend_init)
        print("加载成功")

    def _build_system_prompt(self, user_content: str) -> tuple[str, str]:
        """构建带长期记忆的 system prompt"""
        memory_context = self.memory.build_memory_context(query=user_content, long_term_limit=10)
        # 改进版 —— 强制模型先思考再回答
        SYSTEM_PROMPT = (
            "You are a precise and thorough AI assistant.\n"
            "Rules:\n"
            "- Never fabricate facts. If you don't know, say so.\n"
            "- Be complete, don't truncate explanations.\n"
            "- Address each part of multi-part questions.\n"
            "- Prefer concrete examples over abstract descriptions."
        )
        return SYSTEM_PROMPT, memory_context

    async def _classify_intent(self, text: str) -> tuple[dict, str]:
        _INTENT_RULES = [
            ({"写", "故事", "小说", "write", "story", "creative"}, "story"),
            ({"代码", "code", "function", "def ", "class ", "bug", "debug"}, "code"),
            ({"解释", "explain", "什么是", "原理", "how does"}, "explain"),
            ({"比较", "compare", "区别", "vs", "versus", "差异"}, "compare"),
            ({"分析", "analyze", "analyse", "评估", "assess"}, "analyze"),
            ({"计划", "plan", "步骤", "roadmap", "schedule"}, "plan"),
        ]

        _PARAM_PRESETS = {
            "story":   {"temperature": 0.7,  "top_p": 0.95, "repeat_penalty": 1.1},
            "explain": {"temperature": 0.1,  "top_p": 0.9,  "repeat_penalty": 1.0},
            "compare": {"temperature": 0.1,  "top_p": 0.9,  "repeat_penalty": 1.0},
            "code":    {"temperature": 0.05, "top_p": 0.9,  "repeat_penalty": 1.0},
            "analyze": {"temperature": 0.1,  "top_p": 0.9,  "repeat_penalty": 1.0},
            "plan":    {"temperature": 0.2,  "top_p": 0.9,  "repeat_penalty": 1.05},
            "default": {"temperature": 0.6,  "top_p": 0.9,  "repeat_penalty": 1.05},
        }
        text_lower = text.lower()
        for keywords, label in _INTENT_RULES:
            if any(kw in text_lower for kw in keywords):
                return _PARAM_PRESETS[label], label
        return _PARAM_PRESETS["default"], "default"

    async def generate(self):
        try:
            system_prompt, memory_context = self._build_system_prompt(self.prompt)
            if memory_context:
                system_prompt += f"\n\n{memory_context}"

            user_message = self.prompt
            params, text_type  = await self._classify_intent(self.prompt)
            if text_type != "default":
                user_message = f"{self.prompt}\n\nPlease think step by step before giving your final answer."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ]

            output = self.pipe.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                stream=False,
                **params,
            )

            response_text = output['choices'][0]['message']['content'].strip()
            self.memory.add_turn(self.prompt, response_text)
            print(f"✅ Llama 生成完成（长度：{len(response_text)}）")
            return response_text

        except Exception as e:
            print(f"❌ Llama 生成失败: {e}")
            return []

    async def generate_stream(self):
        if self.model_extra.get("think", "") == "true":
            is_think = True
        else:
            is_think = False

        system_prompt, memory_context = self._build_system_prompt(self.prompt)
        if memory_context:
            system_prompt += f"\n\n{memory_context}"

        user_message = self.prompt + "\n/think" if is_think else self.prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        params, _ = await self._classify_intent(self.prompt)

        stream = self.pipe.create_chat_completion(
            messages=messages,
            max_tokens=4096,
            stream=True,
            **params,
        )

        if is_think:
            async for item in self.think_stream(stream):
                yield item
        else:
            async for item in self.normal_stream(stream):
                yield item

    async def normal_stream(self, stream):
        loop = asyncio.get_event_loop()
        answer_text = ""
        buffer = ""
        print("============ 开始回答 =====================")

        while True:
            chunk = await loop.run_in_executor(None, next, stream, None)
            if chunk is None:
                break
            token = chunk["choices"][0]["delta"].get("content", "")
            if not token:
                continue
            # print("backend:", token)
            answer_text += token
            yield {"type": "text", "text": token}

        if answer_text.strip():
            self.memory.add_turn(self.prompt, answer_text.strip())
            print(f"✅ 生成完成 answer:{len(answer_text)}）")
        yield {"type": "done", "is_think": False}

    async def think_stream(self, stream):
        loop = asyncio.get_event_loop()
        think_text = ""
        answer_text = ""
        buffer = ""
        in_thinking = None
        print("============ 开始思考 =====================")
        while True:
            chunk = await loop.run_in_executor(None, next, stream, None)
            if chunk is None:
                break

            token = chunk["choices"][0]["delta"].get("content", "")
            if not token:
                continue
            buffer += token

            if in_thinking is None:
                if "<think>" in buffer:
                    in_thinking = True
                    remainder = buffer.split("<think>", 1)[1]
                    buffer = ""
                    if remainder:
                        think_text += remainder
                        yield {"type": "thinking", "text": remainder}
                elif len(buffer) > 30:
                    in_thinking = True
                    think_text += buffer
                    yield {"type": "thinking", "text": buffer}
                continue

            if in_thinking:
                if "</think>" in buffer:
                    parts = buffer.split("</think>", 1)
                    if parts[0]:
                        think_text += parts[0]
                        yield {"type": "thinking", "text": parts[0]}
                    in_thinking = False
                    buffer = parts[1]
                    if buffer:
                        answer_text += buffer
                        yield {"type": "text", "text": buffer}
                        buffer = ""
                else:
                    think_text += buffer
                    yield {"type": "thinking", "text": buffer}
                    buffer = ""
                continue

            answer_text += buffer
            yield {"type": "text", "text": buffer}
            buffer = ""

        if buffer:
            if in_thinking:
                think_text += buffer
                yield {"type": "thinking", "text": buffer}
            else:
                answer_text += buffer
                yield {"type": "text", "text": buffer}

        if answer_text.strip():
            self.memory.add_turn(self.prompt, answer_text.strip())
        print(f"✅ 生成完成（think:{len(think_text)} answer:{len(answer_text)}）")
        yield {"type": "done", "is_think": True}

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.prompt = raw.get("content", "")

        self.max_tokens = raw.get("max_tokens", "")
        self.temperature = raw.get("temperature", 0.0)
        self.top_p = raw.get("top_p", 0.9)
        self.repeat_penalty = raw.get("repeat_penalty", 1.05)

    def switch_memory(self, user_id, session_id):
        self.memory = ConversationMemory.get_instance(session_id=session_id,
                                                      user_id=user_id,
                                                      summarizer_fn=self._summarize)
