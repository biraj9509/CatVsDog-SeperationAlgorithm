import cv2
import imutils
def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a color histogram from the HSV color

    color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )

    hist = cv2.calcHist([color], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


def extraRawPixel(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
    return cv2.resize(image, size).flatten()

