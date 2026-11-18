FROM python:3.12-slim

# Hugging Face Spaces ejecuta el contenedor con el usuario 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    NLTK_DATA=/home/user/nltk_data \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user app.py .

# Se hornean datos y recursos NLTK en la imagen para que el arranque no dependa de la red
RUN python -c "from src.data import download_all; download_all()" \
 && python -c "from src.preprocessing import setup_nltk; setup_nltk()"

EXPOSE 7860
# Render inyecta $PORT; en Hugging Face Spaces se usa 7860 por defecto
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
