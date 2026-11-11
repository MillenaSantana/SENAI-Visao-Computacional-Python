# =============================================================
#  AULA 8 - VISAO COMPUTACIONAL | COMO O COMPUTADOR VE
#  Curso: Programacao em IA Generativa - SENAI Frederico Jacob
# -------------------------------------------------------------
#  Mostra que, para o computador, uma imagem e apenas uma grade
#  de numeros. Abre a imagem, mostra o tamanho, converte para
#  tons de cinza e espia um pedacinho da "tabela de pixels".
#
#  Para rodar:  pip install opencv-python numpy
#  Troque 'foto.png' por uma imagem sua (na mesma pasta).
# =============================================================
import cv2

CAMINHO = 'foto.png'
imagem = cv2.imread(CAMINHO)
if imagem is None:
    print(f"Nao encontrei '{CAMINHO}'. Coloque uma imagem com esse nome na pasta.")
else:
    # 1) A imagem e uma tabela de numeros: (altura, largura, 3 cores)
    print("Tamanho da imagem:", imagem.shape)

    # 2) Converte para tons de cinza (1 numero por pixel, de 0 a 255)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('saida_cinza.png', cinza)
    print("Versao em cinza salva em 'saida_cinza.png'.")

    # 3) Espia um pedacinho da grade de numeros (5x5 pixels do canto)
    print("\nUm pedacinho da imagem, em numeros:")
    print(cinza[0:5, 0:5])

# DESAFIO: some 60 a todos os pixels (imagem + 60) e salve o resultado.
# O que acontece com o brilho da imagem?
