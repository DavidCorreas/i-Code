from paddleocr import PaddleOCR


def initialize_ocr(lang="en"):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    return ocr

def ocr(ocr_object: PaddleOCR, image):
    return ocr_object.ocr(image, cls=False)


