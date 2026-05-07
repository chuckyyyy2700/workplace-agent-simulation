"""
ollama_client.py
Windows版 Ollama用クライアント（JSON抽出対応版）
"""
import requests
import json
import re
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:7b",
                 temperature=0.8, max_tokens=512):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str):
        """
        プロンプトを送信し、JSON形式の返答をdictで返す。
        JSONが取得できない場合は空のdictを返す。
        """
        url = f"{self.base_url}/api/generate"

        # JSONで返すよう明示的に指示を追加
        full_prompt = prompt + "\n\n必ず上記のJSON形式のみで返答してください。説明文・前置き・コードブロックは不要です。"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",  # Ollamaのjsonモード
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")

            # まず直接JSONパースを試みる
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # ```json ... ``` ブロックを探す
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            # { } で囲まれた部分を探す
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            # どれも失敗した場合は空dictを返す
            logger.warning(f"JSONパース失敗。生テキスト: {raw[:100]}")
            return {}

        except Exception as e:
            logger.error(f"Ollama APIエラー: {e}")
            return {}

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception:
            logger.error("Ollamaサーバーに接続できません。'ollama serve'を実行してください。")
            return False
