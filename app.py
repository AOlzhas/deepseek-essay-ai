import sys
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# --- Проверка и установка NumPy ---
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} установлен")
except ImportError:
    print("Устанавливаем NumPy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

# --- Загрузка модели с обработкой ошибок ---
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",
            torch_dtype="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        return pipe
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

# --- Генерация ответа ---
def generate_response(prompt, grade):
    if not model:
        return "Модель не загружена. Проверьте ошибки выше."
    
    try:
        instruction = f"""
        [Роль] Учитель для {grade} класса
        [Задача] {prompt}
        [Ответ]:"""
        
        output = model(
            instruction,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        return output[0]['generated_text'].split("[Ответ]:")[-1].strip()
    except Exception as e:
        return f"Ошибка генерации: {str(e)}"

# --- Интерфейс Gradio ---
with gr.Blocks() as app:
    gr.Markdown("# 🎓 Развитие критического мышления")
    
    with gr.Row():
        input_text = gr.Textbox(label="Вопрос", lines=3)
        grade = gr.Slider(6, 11, value=8, label="Класс")
    
    btn = gr.Button("Сгенерировать ответ")
    output = gr.Textbox(label="Ответ", interactive=False)
    
    btn.click(
        fn=generate_response,
        inputs=[input_text, grade],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")