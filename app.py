import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline
import hashlib
import time

# Стили CSS для красивого интерфейса
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.tab {
    background: #f5f7ff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
button {
    background: #6e8efb !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    transition: all 0.3s !important;
}
button:hover {
    background: #5a7de8 !important;
    transform: translateY(-2px);
}
.progress-bar {
    height: 10px;
    background: #e0e0e0;
    border-radius: 5px;
    margin: 10px 0;
}
.progress-fill {
    height: 100%;
    border-radius: 5px;
    background: linear-gradient(90deg, #6e8efb, #a777e3);
}
"""

class EducationPlatform:
    def __init__(self):
        self.users = pd.DataFrame(columns=['user_id', 'login', 'password_hash', 'role', 'full_name', 'email'])
        self.submissions = pd.DataFrame(columns=[
            'student_id', 'teacher_id', 'timestamp', 'essay',
            'argument', 'logic', 'clarity', 'originality', 'feedback'
        ])
        self.evaluator = pipeline("text-classification", model="cointegrated/rubert-tiny2", device="cpu")
        
        # Добавляем тестового преподавателя
        self._add_user("t001", "teacher", self._hash_password("12345"), "teacher", "Иванова Мария", "teacher@school.ru")

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def _add_user(self, user_id, login, password_hash, role, full_name="", email=""):
        new_user = {
            'user_id': user_id,
            'login': login,
            'password_hash': password_hash,
            'role': role,
            'full_name': full_name,
            'email': email
        }
        self.users = pd.concat([self.users, pd.DataFrame([new_user])], ignore_index=True)

    def register_user(self, login, password, role, full_name, email):
        if login in self.users['login'].values:
            return False, "Логин уже занят"
        
        user_id = f"{role[0]}{int(time.time())}"
        self._add_user(user_id, login, self._hash_password(password), role.lower(), full_name, email)
        return True, "Регистрация успешна"

    def authenticate(self, login, password):
        user = self.users[self.users['login'] == login]
        if not user.empty and user.iloc[0]['password_hash'] == self._hash_password(password):
            return "Успешный вход!", True, user.iloc[0]['user_id'], user.iloc[0]['role'], user.iloc[0]['full_name']
        return "Неверный логин или пароль", False, "", "", ""

    def add_submission(self, student_id, teacher_id, essay_text):
        if self._is_duplicate(student_id, essay_text):
            return False
        
        scores, feedback = self.evaluate_essay(essay_text)
        new_row = {
            'student_id': student_id,
            'teacher_id': teacher_id,
            'timestamp': datetime.now(),
            'essay': essay_text,
            **scores,
            'feedback': feedback
        }
        self.submissions = pd.concat([self.submissions, pd.DataFrame([new_row])], ignore_index=True)
        return True

    def _is_duplicate(self, student_id, essay_text):
        return any((self.submissions['student_id'] == student_id) & 
                  (self.submissions['essay'] == essay_text))

    def evaluate_essay(self, essay_text):
        criteria = {
            'argument': "Оцени аргументацию (1-5):",
            'logic': "Оцени логику (1-5):",
            'clarity': "Оцени ясность (1-5):",
            'originality': "Оцени оригинальность (1-5):"
        }
        
        scores = {}
        feedback_parts = []
        
        for criterion, prompt_start in criteria.items():
            full_prompt = f"{prompt_start} {essay_text[:500]}"
            result = self.evaluator(full_prompt)[0]
            score = min(5, max(1, round(float(result['score']) * 5, 1)))
            scores[criterion] = score
            feedback_parts.append(f"{criterion.capitalize()}: {score}/5.0")
        
        feedback = "\n".join(feedback_parts)
        return scores, feedback

    def get_student_progress(self, student_id):
        student_data = self.submissions[self.submissions['student_id'] == student_id]
        if student_data.empty:
            return None, None
        
        avg_scores = student_data[['argument', 'logic', 'clarity', 'originality']].mean()
        return student_data, avg_scores

platform = EducationPlatform()

def register_new_user(login, password, role, full_name, email):
    success, message = platform.register_user(login, password, role, full_name, email)
    return message

def handle_login(input_login, input_password):
    message, success, user_id, role, full_name = platform.authenticate(input_login, input_password)
    if success:
        panels = {
            "user_panel": gr.update(visible=True),
            "student_panel": gr.update(visible=role == "student"),
            "teacher_panel": gr.update(visible=role == "teacher")
        }
        return (
            message, 
            gr.update(value=f"{full_name} ({role.upper()})", visible=True),
            *panels.values()
        )
    return message, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def submit_essay(student_id, teacher_id, essay_text):
    if not all([student_id.strip(), teacher_id.strip(), essay_text.strip()]):
        return None, "Заполните все поля"
    
    if platform.add_submission(student_id, teacher_id, essay_text):
        latest = platform.submissions.iloc[-1]
        scores = {
            'Аргументация': latest['argument'],
            'Логика': latest['logic'],
            'Ясность': latest['clarity'],
            'Оригинальность': latest['originality']
        }
        return scores, latest['feedback']
    return None, "Ошибка при оценке работы"

def update_progress_plot(student_id):
    student_data, avg_scores = platform.get_student_progress(student_id)
    if student_data is None:
        return None
    
    plt.figure(figsize=(10, 5))
    for criterion in ['argument', 'logic', 'clarity', 'originality']:
        plt.plot(student_data['timestamp'], student_data[criterion], label=criterion.capitalize(), marker='o')
    
    plt.title("Прогресс студента")
    plt.ylabel("Оценка (1-5)")
    plt.ylim(1, 5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

with gr.Blocks(css=css) as app:
    with gr.Column():
        gr.Markdown("""<div class="header">
            <h1>🎓 Платформа для развития критического мышления</h1>
            <p>Анализ эссе с использованием ИИ</p>
        </div>""")
        
        with gr.Tabs():
            with gr.TabItem("Вход", id="login"):
                with gr.Column():
                    gr.Markdown("### Вход в систему")
                    with gr.Row():
                        login_input = gr.Textbox(label="Логин", placeholder="Введите ваш логин")
                        password_input = gr.Textbox(label="Пароль", type="password", placeholder="Введите пароль")
                    login_btn = gr.Button("Войти")
                    login_status = gr.Textbox(label="Статус", interactive=False)
                    
            with gr.TabItem("Регистрация", id="register"):
                with gr.Column():
                    gr.Markdown("### Создание учетной записи")
                    with gr.Row():
                        new_login = gr.Textbox(label="Логин")
                        new_password = gr.Textbox(label="Пароль", type="password")
                    with gr.Row():
                        new_full_name = gr.Textbox(label="ФИО")
                        new_email = gr.Textbox(label="Email")
                    role_radio = gr.Radio(["student", "teacher"], label="Роль")
                    register_btn = gr.Button("Зарегистрироваться")
                    register_status = gr.Textbox(label="Статус регистрации", interactive=False)
        
        with gr.Column(visible=False) as user_panel:
            user_info = gr.Textbox(label="Ваш профиль", interactive=False)
            logout_btn = gr.Button("Выйти")
        
        with gr.Column(visible=False) as student_panel:
            with gr.Tab("Отправить эссе"):
                gr.Markdown("### Новая работа")
                teacher_id_input = gr.Textbox(label="ID Преподавателя")
                essay_input = gr.Textbox(label="Текст эссе", lines=10, placeholder="Введите ваше эссе здесь...")
                submit_btn = gr.Button("Отправить на оценку")
                
                with gr.Row():
                    scores_output = gr.Label(label="Результаты оценки")
                    feedback_output = gr.Textbox(label="Комментарии", interactive=False, lines=6)
            
            with gr.Tab("Мой прогресс"):
                progress_plot = gr.Plot(label="История оценок")
                with gr.Row():
                    avg_argument = gr.Number(label="Средняя аргументация")
                    avg_logic = gr.Number(label="Средняя логика")
                    avg_clarity = gr.Number(label="Средняя ясность")
                    avg_originality = gr.Number(label="Средняя оригинальность")
        
        with gr.Column(visible=False) as teacher_panel:
            gr.Markdown("### Панель преподавателя")
            with gr.Tab("Оценить работы"):
                student_select = gr.Dropdown(label="Выберите студента")
                essay_display = gr.Textbox(label="Эссе", interactive=False, lines=10)
                score_argument = gr.Slider(1, 5, step=0.1, label="Аргументация")
                score_logic = gr.Slider(1, 5, step=0.1, label="Логика")
                score_clarity = gr.Slider(1, 5, step=0.1, label="Ясность")
                score_originality = gr.Slider(1, 5, step=0.1, label="Оригинальность")
                save_btn = gr.Button("Сохранить оценку")
            
            with gr.Tab("Статистика группы"):
                group_stats = gr.Dataframe(label="Результаты группы")
                export_btn = gr.Button("Экспорт в Excel")

    # Обработчики событий
    register_btn.click(
        fn=register_new_user,
        inputs=[new_login, new_password, role_radio, new_full_name, new_email],
        outputs=register_status
    )
    
    login_btn.click(
        fn=handle_login,
        inputs=[login_input, password_input],
        outputs=[login_status, user_info, user_panel, student_panel, teacher_panel]
    )
    
    submit_btn.click(
        fn=submit_essay,
        inputs=[user_info, teacher_id_input, essay_input],
        outputs=[scores_output, feedback_output]
    )
    
    user_info.change(
        fn=update_progress_plot,
        inputs=user_info,
        outputs=progress_plot
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")