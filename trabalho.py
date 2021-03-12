import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = {
    'identity': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
    'edge detection': np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=float),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    'laplacian w/ diagonals': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    'laplacian of gaussian': np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=float),
    'scharr': np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]], dtype=float),
    'sobel edge horizontal': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
    'sobel edge vertical': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    'line detection horizontal': np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    'line detection vertical': np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    'line detection 45°': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float),
    'line detection 135°': np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),
    'box blur': (1/9)*np.ones((3,3), dtype=float),
    'gaussian blur 3x3': (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
    'gaussian blur 5x5': (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
    'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    'unsharp masking': (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

def convolve(im, omega, fft=False):
    M, N = im.shape
    A, B = omega.shape
    a, b = A//2, B//2 # núcleo com dimensões ímpares
    if not fft:
        f = np.array(im, dtype=float)
        g = np.zeros_like(f, dtype=float)
        for x in range(M):
            for y in range(N):
                aux = 0.0
                for dx in range(-a, a+1):
                    for dy in range(-b, b+1):
                        if 0 <= x+dx < M and 0 <= y+dy < N: # ou você pode usar "zero pad" na imagem
                            aux += omega[a-dx, b-dy]*f[x+dx, y+dy]
                g[x, y] = aux
        return g
    else:
        im = np.pad(im, ((0,1), (0,1))) # zero pad últimas linha e coluna
        spi = np.fft.fft2(im)
        spf = np.fft.fft2(omega, s=im.shape)
        g = spi*spf
        f = np.fft.ifft2(g)
        return np.real(f)[1:,1:] # elimina as primeiras linha e coluna

# Cria um objeto de captura de vídeo
# (lê de arquivo ou, com argumento 0, captura da webcam)
cap = cv2.VideoCapture(0)#'suporte/cafe.mp4')
# Checa se a camera estiver aberta, 
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
# Lê o fluxo de video (stream)
while(cap.isOpened()):
    # Captura quadro-a-quadro
    ret, frame = cap.read()
    if ret == True:
        # Faça as operações desejadas com a imagem em `frame`
        # Por exemplo, converta-a em tons de cinza:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # E borre a imagem um pouco:
        conv = np.uint8(np.round(convolve(gray, kernel['sharpen'], fft=True)))
        # Mostre a imagem resultante no próximo quadro
        cv2.imshow('Frame', gray)
        # Ao pressionar Q no teclado, o programa encerra
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Se não houver conexão, encerra
    else: 
        break
# Quando tudo estiver encerrado, libera o objeto de captura de vídeo
cap.release()
# Fecha todas as janelas abertas
cv2.destroyAllWindows()