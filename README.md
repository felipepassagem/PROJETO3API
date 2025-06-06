# 🧠 PROJETO3API — Classificador de Esportes com IA

Este projeto consiste em uma **API em Flask** que utiliza um modelo de deep learning treinado com imagens de 15 esportes diferentes. Um frontend simples em HTML permite que o usuário envie uma imagem para a API e visualize a previsão feita pelo modelo.

---

## ✅ Funcionalidade

- Recebe uma imagem via upload
- Classifica a imagem em uma das 15 categorias esportivas
- Retorna:
  - Índice da classe (`class_index`)
  - Nome da classe (`class_name`)
  - Confiança da predição (`confidence`)

---

## 🧪 Como executar

1. Instale as dependências:

```bash
pip install flask flask-cors tensorflow pillow numpy

2. Rode a API:

python app.py

3. Abra index.html no navegador.

Exemplo de resposta da API
{
  "class_index": 0,
  "class_name": "air hockey",
  "confidence": 0.95
}
