# Visão Computacional com Python | SENAI

Projetos práticos desenvolvidos durante o curso **Programação em Inteligência Artificial Generativa — SENAI Frederico Jacob**, explorando processamento de imagens, detecção de objetos e reconhecimento facial com Python e OpenCV.

## Projetos

| Pasta | Descrição |
|---|---|
| `01-Processamento-de-Imagens` | Lê dimensões e valores dos pixels e converte uma imagem em tons de cinza. |
| `02-Deteccao-de-Objetos` | Usa limiarização e contornos para localizar formas geométricas e desenhar caixas delimitadoras. |
| `03-Deteccao-Facial` | Detecta rostos em fotografias com Haar Cascade (não identifica pessoas). |
| `04-Reconhecimento-Facial` | Cadastra rostos pela webcam, treina um reconhecedor LBPH e exibe nomes ou “Desconhecido”. |

## Tecnologias
Python · OpenCV · NumPy · Haar Cascade · LBPH

## Como executar

1. Instale Python 3 e abra o terminal na pasta principal.
2. Instale as dependências: `pip install -r requirements.txt`.
3. Entre na pasta desejada e execute `python nome_do_arquivo.py`.
4. Para o projeto 01, coloque uma imagem chamada `foto.png` na pasta.
5. Para o projeto 02, a imagem `objetos.png` já está incluída.
6. Para o projeto 03, adicione uma imagem autorizada chamada `foto_turma.jpg`.
7. Para o projeto 04, tenha uma webcam disponível e use o menu: cadastrar → treinar → reconhecer.

## Exemplo de imagem para detecção de objetos

![Formas geométricas utilizadas na atividade](02-Deteccao-de-Objetos/objetos.png)

## Privacidade e limitações

Os exercícios foram desenvolvidos para fins educacionais. O reconhecimento facial utiliza LBPH e pode cometer erros; não deve ser utilizado como mecanismo único de controle de acesso ou autenticação. Utilize apenas imagens e dados biométricos de pessoas que autorizaram seu uso. Não publique fotos de cadastro, modelos treinados nem arquivos com nomes pessoais.

## Contexto
Atividades realizadas no curso do SENAI, organizadas para documentar a aprendizagem e a evolução técnica.
