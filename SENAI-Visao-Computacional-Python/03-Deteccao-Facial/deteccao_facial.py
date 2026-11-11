# =============================================================
#  AULA 8 - VISAO COMPUTACIONAL | DETECCAO DE ROSTOS
#  Curso: Programacao em IA Generativa - SENAI Frederico Jacob
# -------------------------------------------------------------
#  Encontra rostos em uma foto usando o detector Haar, que ja
#  vem pronto no OpenCV, e desenha uma caixa em cada rosto.
#
#  Para rodar:  pip install opencv-python
#  Use uma foto com uma ou mais pessoas (ex.: 'foto_turma.jpg').
#  OBS: isto DETECTA rostos (onde estao), nao RECONHECE quem sao.
# =============================================================
import cv2

CAMINHO = 'foto_turma.jpg'
imagem = cv2.imread(CAMINHO)
if imagem is None:
    print(f"Nao encontrei '{CAMINHO}'. Coloque uma foto com esse nome na pasta.")
    raise SystemExit

# carrega o detector de rostos que ja vem no OpenCV
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# procura rostos de varios tamanhos na imagem
rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5)
print("Rostos encontrados:", len(rostos))

# desenha uma caixa verde em cada rosto
for (x, y, w, h) in rostos:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite('rostos_detectados.jpg', imagem)
print("Resultado salvo em 'rostos_detectados.jpg'.")

# DICA: se ele nao achar todos os rostos, tente minNeighbors=3.
# ETICA: rostos sao dados biometricos. Use fotos com autorizacao (LGPD).
