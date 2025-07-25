from __future__ import annotations

import uuid
import typing as t
from logging import getLogger
from ai_handler.providers.ai_provider_client import AiProviderClient
from ai_handler.providers.ai_provider_client import AIChat
import ai_handler.errors as ex
from enum import Enum

if t.TYPE_CHECKING:
    from google.genai.types import GenerateContentConfig
    from google.genai.chats import Chat as GenaiChat

logger = getLogger("ai_handler")


class GeminiModelType(Enum):
    G2_5_pro = "gemini-2.5-pro"
    G2_5_flash = "gemini-2.5-flash"
    G2_5_flash_lite_preview_06_17 = "gemini-2.5-flash-lite-preview-06-17"
    G2_5_flash_preview_native_audio_dialog = (
        "gemini-2.5-flash-preview-native-audio-dialog"
    )
    G2_5_flash_exp_native_audio_thinking_dialog = (
        "gemini-2.5-flash-exp-native-audio-thinking-dialog"
    )
    G2_5_flash_preview_tts = "gemini-2.5-flash-preview-tts"
    G2_5_pro_preview_tts = "gemini-2.5-pro-preview-tts"
    G2_0_flash = "gemini-2.0-flash"
    G2_0_flash_preview_image_generation = "gemini-2.0-flash-preview-image-generation"
    G2_0_flash_lite = "gemini-2.0-flash-lite"
    G1_5_flash = "gemini-1.5-flash"
    G1_5_flash_8b = "gemini-1.5-flash-8b"
    G1_5_pro = "gemini-1.5-pro"
    GEMINI_EMBEDDING_EXP = "gemini-embedding-exp"
    IMAGEN_4_0_generate_preview_06_06 = "imagen-4.0-generate-preview-06-06"
    IMAGEN_4_0_ultra_generate_preview_06_06 = "imagen-4.0-ultra-generate-preview-06-06"
    IMAGEN_3_0_generate_002 = "imagen-3.0-generate-002"
    VEO_2_0_generate_001 = "veo-2.0-generate-001"
    GEMINI_LIVE_2_5_flash_preview = "gemini-live-2.5-flash-preview"
    G2_0_flash_live_001 = "gemini-2.0-flash-live-001"


class GeminiChat(AIChat):
    def __init__(self, chat_id: str, context: GenaiChat, config: GenerateContentConfig):
        from google.genai.chats import Chat as GenaiChat

        if not isinstance(context, GenaiChat):
            raise TypeError("context must be an instance of google.genai.chats.Chat")
        super().__init__(chat_id)
        self.context = context
        self.config = config

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @chat_id.setter
    def chat_id(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Chat ID must be a string")
        self._chat_id = value

    def ask(
        self,
        prompt: str,
        config: t.Optional[t.Any] = None,
    ) -> str:
        if not self.context:
            raise ex.ClientError("Chat context is not initialized")
        if config is None:
            config = self.config
        response = self.context.send_message(prompt, config=config)
        return response.text


class Gemini(AiProviderClient):
    def __init__(
        self,
        default_model: str | GeminiModelType,
        api_key: str,
        backup_models: t.Optional[list[str | GeminiModelType]] = None,
        default_sys_instructions: t.Optional[str] = None,
        default_temperature: float = 0.2,
        default_limit_tokens: t.Optional[int] = None,
    ):
        self.chats = {}
        if backup_models is None:
            backup_models = []
        self.backup_models = [GeminiModelType(m) for m in backup_models]
        try:
            from google import genai

            self.genai = genai
        except ImportError as e:
            raise ImportError(
                "The Gemini provider requires the 'google' package. "
                "Install it with 'pip install ai_handler[google]'"
            ) from e
        self.default_model = GeminiModelType(default_model)
        self.api_key = api_key
        self.default_sys_instructions = default_sys_instructions or ""
        self.default_limit_tokens = default_limit_tokens
        self.default_temperature = default_temperature
        self.client = genai.Client(api_key=api_key)

    @property
    def chats(self) -> dict[str, AIChat]:
        return self._chats

    @chats.setter
    def chats(self, value: dict[str, AIChat]):
        if not isinstance(value, dict):
            raise TypeError("chats must be a dictionary of AIChat instances")
        self._chats = value

    def get_config(
        self,
        temperature: t.Optional[float] = None,
        system_instructions: t.Optional[str] = None,
        limit_tokens: t.Optional[int] = None,
    ) -> GenerateContentConfig:
        from google.genai.types import GenerateContentConfig
        return GenerateContentConfig(
            temperature=(
                temperature if temperature is not None else self.default_temperature
            ),
            system_instruction=system_instructions or self.default_sys_instructions,
            max_output_tokens=(
                limit_tokens if limit_tokens is not None else self.default_limit_tokens
            ),
        )

    def ask(
        self,
        prompt: str,
        model: t.Optional[GeminiModelType | str] = None,
        temperature: t.Optional[float] = None,
        system_instructions: t.Optional[str] = None,
        limit_tokens: t.Optional[int] = None,
        use_backups: bool = True,
    ) -> str:
        logger.debug(f"Asking Gemini with prompt: {prompt}")
        if model is None:
            model = self.default_model
        elif isinstance(model, str):
            model = GeminiModelType(model)
        if use_backups and self.backup_models:
            models = [model] + [m for m in self.backup_models if m != model]
        else:
            models = [model]
        from google.genai.errors import ServerError
        for model in models:
            try:
                chat = self.create_chat(
                    model=model,
                    temperature=temperature,
                    system_instructions=system_instructions,
                    limit_tokens=limit_tokens,
                )
                return self.ask_chat(prompt, chat)
            except ServerError as e:
                if e.code == 503:  # Service Unavailable
                    logger.warning(f"Model {model} is unavailable, trying next model.")
                    continue
                logger.error(f"Server error with model {model}: {e}")
                raise ex.ProviderError(
                    f"Server error while asking with model {model}: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error with model {model}: {e}")
                raise ex.ProviderError(
                    f"Unexpected error while asking with model {model}: {e}"
                ) from e
        logger.error("All models are unavailable or failed to respond.")
        raise ex.ProviderError("All models are unavailable or failed to respond.")        

    def ask_chat(self, prompt: str, chat: GeminiChat | str) -> str:
        if isinstance(chat, str):
            chat = self.chats.get(chat)
        if not isinstance(chat, GeminiChat):
            raise TypeError(
                f"chat must be an instance of GeminiChat or a chat_id string but got {type(chat)}"
            )
        return chat.ask(prompt)

    
    def create_chat(
        self,
        model: t.Optional[GeminiModelType] = None,
        temperature: t.Optional[float] = None,
        system_instructions: t.Optional[str] = None,
        limit_tokens: t.Optional[int] = None,
    ) -> GeminiChat:
        logger.debug("Creating a new Gemini chat")
        from google.genai.errors import APIError

        try:
            config = self.get_config(
                temperature=temperature,
                system_instructions=system_instructions,
                limit_tokens=limit_tokens,
            )
            if model is None:
                model = self.default_model
            sdk_chat = self.client.chats.create(model=model.value, config=config)
            if not sdk_chat:
                raise ex.ProviderError("Failed to create chat with Gemini provider")
            chat_id = str(uuid.uuid4())
            chat = GeminiChat(chat_id, sdk_chat, config=config)
            self.chats[chat_id] = chat
            return chat
        except APIError as e:
            raise ex.ProviderError(f"API error while creating chat: {e}") from e
        except Exception as e:
            raise ex.ProviderError(f"Unexpected error while creating chat: {e}") from e
