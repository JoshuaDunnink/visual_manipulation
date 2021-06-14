import cv2
from PIL import ImageGrab
import numpy as np
from timer import timer

class CascadeIdentifier:
    def __init__(
        self,
        screen=None,
        haar_cascade="HAAR_cascades\\front_face.xml",
        box_color=(0, 0, 255),
    ):
        if not screen:
            self.screen = ImageGrab.grab()
            self.screen_image = np.array(self.screen)
        else:
            self.screen_image = cv2.imread(screen)
        self.box_color = box_color
        self.gray_image = cv2.cvtColor(self.screen_image, cv2.COLOR_BGR2GRAY)
        self.tracker = cv2.CascadeClassifier(haar_cascade)

    def detect(self):
        self.detect_faces = self.tracker.detectMultiScale(self.gray_image)

        for (x, y, w, h) in self.detect_faces:
            cv2.rectangle(
                self.screen_image, (x, y), (x + w, y + h), self.box_color, 2
            )
            cv2.putText(
                img=self.screen_image,
                text='face',
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=self.box_color,
                thickness=2,
            )

    def show_detections(self):
        cv2.imshow('detections', self.screen_image)
        cv2.waitKey()


@timer
def do_it(face_identifier):
    face_identifier.detect()


def main():
    face_identifier = CascadeIdentifier()
    for _ in range(10):
        do_it(face_identifier)


if __name__ == "__main__":
    main()
