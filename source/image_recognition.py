import cv2


class ImageRecognizor:
    def __init__(self):
        self.method = cv2.TM_SQDIFF_NORMED

    def find_target_on_screen(self, target, screen):
        self.target = cv2.imread(str(target))
        self.screen = cv2.imread(str(screen))

        self.result = cv2.matchTemplate(
            self.target,
            self.screen,
            self.method,
        )

    def show_target_on_screen(self):
        (
            self.x_location,
            self.y_location,
        ) = self._get_x_and_y_location_on_cv2_match()
        self._draw_rectangle_on_screen()

        cv2.imshow("output", self.screen)
        cv2.waitKey(0)

    def get_center_x_and_y_pixel_location(self):
        x, y = self._get_x_and_y_location_on_cv2_match()
        width, height = self._get_target_size()
        x += int(width / 2)
        y += int(height / 2)
        return x, y

    def _get_x_and_y_location_on_cv2_match(self):
        return cv2.minMaxLoc(self.result)[2]

    def _get_target_size(self):
        return self.target.shape[:2]

    def _draw_rectangle_on_screen(self):
        width, height = self._get_target_size()

        cv2.rectangle(
            self.screen,
            (self.x_location, self.y_location),
            (self.x_location + height, self.y_location + width),
            (0, 0, 255),
            2,
        )
