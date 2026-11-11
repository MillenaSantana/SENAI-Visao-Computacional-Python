# =============================================================
#  AULA 8 - VISAO COMPUTACIONAL | DETECCAO DE OBJETOS
#  Curso: Programacao em IA Generativa - SENAI Frederico Jacob
# -------------------------------------------------------------
#  Encontra objetos em uma imagem pelos seus CONTORNOS (as bordas
#  que separam o objeto do fundo) e desenha uma caixa em cada um.
#
#  Para rodar:  pip install opencv-python numpy
#  Use uma imagem com objetos/formas sobre um fundo claro.
# =============================================================
import cv2

CAMINHO = 'objetos.png'
imagem = cv2.imread(CAMINHO)
if imagem is None:
    print(f"Nao encontrei '{CAMINHO}'. Coloque uma imagem com esse nome na pasta.")
    raise SystemExit

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# separa os objetos (mais escuros) do fundo (mais claro)
limiar = cv2.threshold(cinza, 240, 255, cv2.THRESH_BINARY_INV)[1]

# encontra os contornos dos objetos
contornos, _ = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# desenha uma caixa ao redor de cada objeto encontrado
n = 0
for c in contornos:
    if cv2.contourArea(c) > 300:          # ignora ruidos pequenos
        n += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(imagem, f"obj {n}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

print("Objetos detectados:", n)
cv2.imwrite('objetos_detectados.png', imagem)
print("Resultado salvo em 'objetos_detectados.png'.")

# DESAFIO: mude o valor 300 para 2000 e veja quais objetos deixam de ser contados.
