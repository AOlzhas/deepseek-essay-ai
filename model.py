# model.py - Исправленная версия с правильной загрузкой модели
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class CriticalThinkingModel:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Проверка доступных устройств
        self.use_cuda = torch.cuda.is_available()
        self.use_mps = torch.backends.mps.is_available() if torch.backends.mps.is_built() else False
        
        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Определение параметров загрузки
        torch_dtype = torch.float16 if self.use_cuda else torch.float32
        device_map = "auto"  # Позволяет accelerate самому выбрать устройство
        
        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # Создание конвейера БЕЗ указания устройства
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_response(self, prompt: str, grade: int) -> str:
        """Генерация образовательного ответа"""
        instruction = f"""
        [Роль] Ты - учитель по критическому мышлению для {grade} класса.
        [Задача] {prompt}
        [Требования]:
        1. Разбор проблемы
        2. Вопросы для обсуждения
        3. Рекомендации
        [Ответ]:"""
        
        try:
            output = self.pipe(
                instruction,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return output[0]['generated_text'].split("[Ответ]:")[-1].strip()
        except Exception as e:
            return f"Ошибка генерации: {str(e)}"