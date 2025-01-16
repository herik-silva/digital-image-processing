import cv2
import matplotlib.pyplot as plt
import numpy as np
import typing as _typing

RawImage = _typing.Union[cv2.typing.MatLike, str]
MatLike = cv2.typing.MatLike
Size = cv2.typing.Size

class Image:
    matlike: MatLike

    def __init__(self, image: RawImage):

        if isinstance(image, str):
            self.matlike = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            self.matlike = image
    

    def add_salt_pepper_noise(self, salt_prob: float, pepper_prob: float, image: MatLike=None, seed: int=10):
        np.random.seed(seed)

        if image is None:
            image = self.matlike

        noisy_image = np.copy(image)
        total_pixels = image.size
        
        salt_pixels = int(total_pixels * salt_prob)
        pepper_pixels = int(total_pixels * pepper_prob)

        coords_salt = [np.random.randint(0, i-1, salt_pixels) for i in image.shape]
        noisy_image[coords_salt[0], coords_salt[1]] = 255
        coords_pepper = [np.random.randint(0, i-1, pepper_pixels) for i in image.shape]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0

        return noisy_image
    
    def gaussian(self, image: MatLike=None, ksize: Size=(5,5), sigmaX: float=0):
        if image is None:
            return cv2.GaussianBlur(self.matlike, ksize, sigmaX)
        
        return cv2.GaussianBlur(noisy_image, ksize, sigmaX)
    
    def median(self, image:MatLike=None, ksize: int=5):
        if image is None:
            return cv2.medianBlur(self.matlike, ksize)
        
        return cv2.medianBlur(image, ksize)
    
    def mean(self, image: MatLike=None, ksize: Size=(5,5)):
        if image is None:
            return cv2.blur(self.matlike, ksize)
        
        return cv2.blur(image, ksize)
    
    def otsu(self, type: int, image: MatLike=None, tresh: float=0, maxval: float=255):
        if image is None:
            return cv2.threshold(self.image, tresh, maxval, type)
        
        return cv2.threshold(image, tresh, maxval, type)


def save_plot(image: Image,  main_title: str, file_name: str, other_image: Image = None, other_title: str=""):
    plt.clf()
    if other_image is not None:
        plt.subplot(1, 2, 1)
        
    plt.title(main_title)
    plt.imshow(image, cmap='gray')

    if other_image is not None:
        if not other_title:
            raise ValueError("other_title is missing")
        
        plt.subplot(1, 2, 2)
        plt.title(other_title)
        plt.imshow(other_image, cmap='gray')

    plt.savefig(f'{file_name}.png')


image = Image("./fruits/test/orange_94.jpg")
noisy_image = image.add_salt_pepper_noise(0.02, 0.02, seed=5)
save_plot(image.matlike, "Original", "aplica_ruido", noisy_image, "Com ruído")

gaussean_image = image.gaussian(noisy_image)
save_plot(noisy_image, "Imagem com ruído", "ruido_gausseano", gaussean_image, "Imagem com filtro gausseano")

median_image = image.median(noisy_image)
save_plot(noisy_image, "Imagem com ruído", "ruido_mediana", median_image, "Imagem com filtro mediana")

mean_image = image.mean(noisy_image)
save_plot(noisy_image, "Imagem com ruído", "ruido_media", mean_image, "Imagem com filtro de média")

otsu_limiar, otsu_segmented = image.otsu(cv2.THRESH_BINARY + cv2.THRESH_OTSU, median_image)
save_plot(median_image, "Filtro Mediana", "mediana_otsu_segmentado", otsu_segmented, "Segmentada com Otsu")
print(f"Limiar escolhido pelo método de Otsu(Média): {otsu_limiar}")

otsu_limiar, otsu_segmented = image.otsu(cv2.THRESH_BINARY + cv2.THRESH_OTSU, gaussean_image)
save_plot(gaussean_image, "Filtro Gausseano", "gausseana_otsu_segmentado", otsu_segmented, "Segmentada com Otsu")
print(f"Limiar escolhido pelo método de Otsu(Gausseano): {otsu_limiar}")

otsu_limiar, otsu_segmented = image.otsu(cv2.THRESH_BINARY + cv2.THRESH_OTSU, gaussean_image)
save_plot(mean_image, "Filtro Média", "media_otsu_segmentado", otsu_segmented, "Segmentada com Otsu")
print(f"Limiar escolhido pelo método de Otsu(Média): {otsu_limiar}")