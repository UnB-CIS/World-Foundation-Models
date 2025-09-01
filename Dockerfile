FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir economiza espaço na imagem final
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os códigos do projeto para o diretório de trabalho /app
COPY . .

EXPOSE 8888

# - '--ip=0.0.0.0' permite que ele seja acessado de fora do container
# - '--port=8888' especifica a porta
# - '--allow-root' é necessário pois o Docker executa comandos como usuário root por padrão
# - '--no-browser' impede que ele tente abrir um navegador dentro do container
# - '--NotebookApp.token=''' desativa a necessidade de token para simplificar 
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
