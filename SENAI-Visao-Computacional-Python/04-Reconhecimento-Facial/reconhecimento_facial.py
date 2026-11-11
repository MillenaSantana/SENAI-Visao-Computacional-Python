# =============================================================
#  AULA 8 - VISAO COMPUTACIONAL | RECONHECIMENTO FACIAL (WEBCAM)
#  Curso: Programacao em IA Generativa - SENAI Frederico Jacob
# -------------------------------------------------------------
#  Agora o programa nao so DETECTA rostos: ele aprende quem e
#  cada pessoa e RECONHECE pelo nome, em tempo real, na webcam.
#
#  Funciona em 3 etapas (menu):
#    1) CADASTRAR  - tira ~40 fotos do rosto de uma pessoa
#    2) TREINAR    - o OpenCV aprende com as fotos cadastradas
#    3) RECONHECER - a webcam mostra o nome de quem aparece
#
#  Para rodar:  pip install opencv-contrib-python numpy
#  (atencao: e o opencv-CONTRIB, que tem o modulo cv2.face;
#   se ja tiver o opencv-python instalado, desinstale antes:
#   pip uninstall opencv-python)
#
#  Tecnica usada: LBPH (Local Binary Patterns Histograms), um
#  metodo classico e leve, bom para aprender o conceito.
#  Em tempo real, quanto MENOR a "distancia", mais parecido.
# =============================================================
import json
import os
import unicodedata

import cv2
import numpy as np

# ---------------- configuracoes ----------------
PASTA_ROSTOS = 'rostos_cadastrados'   # uma subpasta por pessoa
ARQ_MODELO = 'modelo.yml'             # modelo treinado
ARQ_NOMES = 'nomes.json'              # rotulo -> nome da pessoa
FOTOS_POR_PESSOA = 40                 # quantas fotos tirar no cadastro
TAMANHO = (200, 200)                  # tamanho padrao dos rostos
LIMIAR = 70                           # abaixo disso = reconhecido
                                      # (menor = mais rigoroso)

if not hasattr(cv2, 'face'):
    print("Modulo cv2.face nao encontrado.")
    print("Instale com:  pip uninstall opencv-python")
    print("              pip install opencv-contrib-python")
    raise SystemExit

try:
    ARQ_HAAR = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
except AttributeError:
    ARQ_HAAR = ''
detector = cv2.CascadeClassifier(ARQ_HAAR)
if detector.empty():
    print("Nao consegui carregar o detector de rostos (Haar).")
    print(f"Versao do OpenCV instalada: {cv2.__version__}")
    print("Isso acontece no OpenCV 5.x. Instale a versao 4.x:")
    print("  pip uninstall opencv-python opencv-contrib-python")
    print('  pip install "opencv-contrib-python<5" numpy')
    raise SystemExit


# ---------------- funcoes auxiliares ----------------
def nome_seguro(texto):
    """Tira acentos e troca espacos por _ (nome de pasta seguro)."""
    sem_acento = unicodedata.normalize('NFKD', texto)
    sem_acento = sem_acento.encode('ascii', 'ignore').decode('ascii')
    return sem_acento.strip().replace(' ', '_')


def abrir_camera():
    camera = cv2.VideoCapture(0)  # tente 1 se tiver mais de uma camera
    if not camera.isOpened():
        print("Nao consegui abrir a webcam. Verifique se ela esta")
        print("conectada e se nenhum outro programa esta usando.")
        return None
    return camera


def preparar_rosto(cinza, caixa):
    """Recorta o rosto, padroniza o tamanho e equaliza o contraste.
    Usado no cadastro, no treino e no reconhecimento (mesmo preparo)."""
    x, y, w, h = caixa
    rosto = cv2.resize(cinza[y:y + h, x:x + w], TAMANHO)
    return cv2.equalizeHist(rosto)


def detectar(cinza):
    return detector.detectMultiScale(
        cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))


# ---------------- 1) CADASTRAR ----------------
def cadastrar():
    nome = input("Nome da pessoa: ").strip()
    if not nome:
        print("Nome vazio. Cancelado.")
        return

    # ETICA / LGPD: rosto e dado biometrico, so com autorizacao
    resp = input(f"{nome} autorizou o uso do rosto para esta atividade? (s/n): ")
    if resp.strip().lower() != 's':
        print("Sem autorizacao, nao e possivel cadastrar. Cancelado.")
        return

    pasta = os.path.join(PASTA_ROSTOS, nome_seguro(nome))
    os.makedirs(pasta, exist_ok=True)
    existentes = len(os.listdir(pasta))

    camera = abrir_camera()
    if camera is None:
        return

    print("Olhe para a camera e mexa a cabeca devagar (de frente,")
    print("um pouco para os lados, com e sem sorriso).")
    print("Aperte Q para cancelar.")

    capturadas = 0
    quadros_com_rosto = 0
    while capturadas < FOTOS_POR_PESSOA:
        ok, quadro = camera.read()
        if not ok:
            break
        quadro = cv2.flip(quadro, 1)
        cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        rostos = detectar(cinza)

        # so captura quando ha exatamente UM rosto na imagem
        if len(rostos) == 1:
            x, y, w, h = rostos[0]
            cv2.rectangle(quadro, (x, y), (x + w, y + h), (0, 255, 0), 2)
            quadros_com_rosto += 1
            if quadros_com_rosto % 5 == 0:  # 1 foto a cada 5 quadros
                rosto = preparar_rosto(cinza, rostos[0])
                numero = existentes + capturadas + 1
                cv2.imwrite(os.path.join(pasta, f'{numero:03d}.jpg'), rosto)
                capturadas += 1
        else:
            cv2.putText(quadro, "Deixe so 1 rosto visivel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(quadro, f"Fotos: {capturadas}/{FOTOS_POR_PESSOA}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Cadastro - Q para cancelar', quadro)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"{capturadas} fotos salvas para '{nome}'.")
    print("Agora use a opcao 2 (TREINAR) para o programa aprender.")


# ---------------- 2) TREINAR ----------------
def treinar():
    if not os.path.isdir(PASTA_ROSTOS):
        print("Nenhuma pessoa cadastrada ainda. Use a opcao 1.")
        return

    imagens, rotulos, nomes = [], [], []
    for pasta in sorted(os.listdir(PASTA_ROSTOS)):
        caminho = os.path.join(PASTA_ROSTOS, pasta)
        if not os.path.isdir(caminho):
            continue
        rotulo = len(nomes)
        fotos = 0
        for arquivo in sorted(os.listdir(caminho)):
            img = cv2.imread(os.path.join(caminho, arquivo),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imagens.append(cv2.resize(img, TAMANHO))
            rotulos.append(rotulo)
            fotos += 1
        if fotos > 0:
            nomes.append(pasta.replace('_', ' '))
            print(f"  {pasta.replace('_', ' ')}: {fotos} fotos")

    if not imagens:
        print("Nao encontrei fotos. Use a opcao 1 primeiro.")
        return
    if len(nomes) < 2:
        print("Aviso: com so 1 pessoa cadastrada o resultado fica menos")
        print("confiavel. Ideal: cadastrar 2 ou mais pessoas.")

    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.train(imagens, np.array(rotulos))
    reconhecedor.write(ARQ_MODELO)
    with open(ARQ_NOMES, 'w', encoding='utf-8') as f:
        json.dump(nomes, f, ensure_ascii=False)
    print(f"Treino concluido com {len(nomes)} pessoa(s). Modelo salvo.")


# ---------------- 3) RECONHECER ----------------
def reconhecer():
    if not (os.path.exists(ARQ_MODELO) and os.path.exists(ARQ_NOMES)):
        print("Modelo nao encontrado. Faca o cadastro (1) e o treino (2).")
        return

    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read(ARQ_MODELO)
    with open(ARQ_NOMES, encoding='utf-8') as f:
        nomes = json.load(f)

    camera = abrir_camera()
    if camera is None:
        return
    print("Reconhecendo... Q ou ESC para voltar ao menu.")

    while True:
        ok, quadro = camera.read()
        if not ok:
            break
        quadro = cv2.flip(quadro, 1)
        cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)

        for caixa in detectar(cinza):
            x, y, w, h = caixa
            rotulo, distancia = reconhecedor.predict(preparar_rosto(cinza, caixa))

            if distancia < LIMIAR:
                # texto sem acento: o OpenCV nao desenha acentos
                nome = unicodedata.normalize('NFKD', nomes[rotulo])
                nome = nome.encode('ascii', 'ignore').decode('ascii')
                cor = (0, 255, 0)   # verde = reconhecido
            else:
                nome = "Desconhecido"
                cor = (0, 0, 255)   # vermelho = nao reconhecido

            cv2.rectangle(quadro, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(quadro, f"{nome} ({distancia:.0f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        cv2.imshow('Reconhecimento - Q para sair', quadro)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    camera.release()
    cv2.destroyAllWindows()


# ---------------- menu principal ----------------
while True:
    print("\n=== RECONHECIMENTO FACIAL ===")
    print("1 - Cadastrar uma pessoa")
    print("2 - Treinar o reconhecedor")
    print("3 - Reconhecer pela webcam")
    print("0 - Sair")
    opcao = input("Escolha: ").strip()

    if opcao == '1':
        cadastrar()
    elif opcao == '2':
        treinar()
    elif opcao == '3':
        reconhecer()
    elif opcao == '0':
        break
    else:
        print("Opcao invalida.")

# DICA: se reconhecer a pessoa errada, diminua LIMIAR (ex.: 60).
#       se nao reconhece ninguem, aumente (ex.: 80).
# DICA: cadastre com a mesma iluminacao em que vai testar.
# ETICA: rostos sao dados biometricos (LGPD). Cadastre apenas
#        pessoas que autorizaram e apague a pasta
#        'rostos_cadastrados' ao terminar a atividade.