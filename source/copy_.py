import cv2
import numpy as np

DEFAULT_TEMPLATE_MATCHING_THRESHOLD = 0.5


class Template:
    """
    A class defining a template
    """

    def __init__(
        self,
        image_path,
        label,
        color,
        matching_threshold=DEFAULT_TEMPLATE_MATCHING_THRESHOLD,
    ):
        """
        Args:
            image_path (str): path of the template image path
            label (str): the label corresponding to the template
            color (List[int]): the color associated with the label (to plot detections)
            matching_threshold (float): the minimum similarity score to consider an object is detected by template
                matching
        """
        self.image_path = image_path
        self.label = label
        self.color = color
        self.template = cv2.imread(image_path)
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold


class Detector:
    def __init__(self, templates, image):
        self.detections = []
        self.templates = templates
        self.image = cv2.imread(image)
        self.match_method = cv2.TM_CCOEFF_NORMED
        self.NMS_THRESHOLD = 0.2

    def detect(self, template=None):
        if not template:
            for template in self.templates:
                template_matching = cv2.matchTemplate(
                    template.template, self.image, self.match_method
                )

                match_locations = np.where(
                    template_matching >= template.matching_threshold
                )

                for (x, y) in zip(match_locations[1], match_locations[0]):
                    match = {
                        "TOP_LEFT_X": x,
                        "TOP_LEFT_Y": y,
                        "BOTTOM_RIGHT_X": x + template.template_width,
                        "BOTTOM_RIGHT_Y": y + template.template_height,
                        "MATCH_VALUE": template_matching[y, x],
                        "LABEL": template.label,
                        "COLOR": template.color,
                    }

                    self.detections.append(match)

        self.detections = self._non_max_suppression(
            self.detections, non_max_suppression_threshold=self.NMS_THRESHOLD
        )

    def show_detections(self):
        image_with_detections = self.image.copy()
        for detection in self.detections:
            cv2.rectangle(
                image_with_detections,
                (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
                (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
                detection["COLOR"],
                2,
            )
            cv2.putText(
                img=image_with_detections,
                text=str(detection['MATCH_VALUE'])[:4],
                org=(detection["TOP_LEFT_X"] + 2, detection["TOP_LEFT_Y"] + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=detection["COLOR"],
                thickness=2,
                bottomLeftOrigin=cv2.LINE_AA,
            )

        cv2.imshow("result", image_with_detections)
        cv2.waitKey(0)

    def _non_max_suppression(
        self,
        objects,
        non_max_suppression_threshold=0.5,
        score_key="MATCH_VALUE",
    ):
        """
        Filter objects overlapping with IoU over threshold by keeping only the one with maximum score.
        Args:
            objects (List[dict]): a list of objects dictionaries, with:
                {score_key} (float): the object score
                {top_left_x} (float): the top-left x-axis coordinate of the object bounding box
                {top_left_y} (float): the top-left y-axis coordinate of the object bounding box
                {bottom_right_x} (float): the bottom-right x-axis coordinate of the object bounding box
                {bottom_right_y} (float): the bottom-right y-axis coordinate of the object bounding box
            non_max_suppression_threshold (float): the minimum IoU value used to filter overlapping boxes when
                conducting non max suppression.
            score_key (str): score key in objects dicts
        Returns:
            List[dict]: the filtered list of dictionaries.
        """
        sorted_objects = sorted(
            objects, key=lambda obj: obj[score_key], reverse=True
        )
        filtered_objects = []
        for object_ in sorted_objects:
            overlap_found = False
            for filtered_object in filtered_objects:
                iou = self._get_iou(object_, filtered_object)
                if iou > non_max_suppression_threshold:
                    overlap_found = True
                    break
            if not overlap_found:
                filtered_objects.append(object_)
        return filtered_objects

    @staticmethod
    def _get_iou(a, b, epsilon=1e-5):
        """Given two boxes `a` and `b` defined as a list of four numbers:
                [x1,y1,x2,y2]
            where:
                x1,y1 represent the upper left corner
                x2,y2 represent the lower right corner
            It returns the Intersect of Union score for these two boxes.

        Args:
            a:          (list of 4 numbers) [x1,y1,x2,y2]
            b:          (list of 4 numbers) [x1,y1,x2,y2]
            epsilon:    (float) Small value to prevent division by zero

        Returns:
            (float) The Intersect of Union score.
        """

        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a["TOP_LEFT_X"], b["TOP_LEFT_X"])
        y1 = max(a["TOP_LEFT_Y"], b["TOP_LEFT_Y"])
        x2 = min(a["BOTTOM_RIGHT_X"], b["BOTTOM_RIGHT_X"])
        y2 = min(a["BOTTOM_RIGHT_Y"], b["BOTTOM_RIGHT_Y"])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = x2 - x1
        height = y2 - y1
        # handle case where there is NO overlap
        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a["BOTTOM_RIGHT_X"] - a["TOP_LEFT_X"]) * (
            a["BOTTOM_RIGHT_Y"] - a["TOP_LEFT_Y"]
        )
        area_b = (b["BOTTOM_RIGHT_X"] - b["TOP_LEFT_X"]) * (
            b["BOTTOM_RIGHT_Y"] - b["TOP_LEFT_Y"]
        )
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined + epsilon)
        return iou


templates = [
    Template(image_path="compare/waldo.png", label="1", color=(0, 0, 255)),
    Template(
        image_path="compare/blond_on_beach.png",
        label="2",
        color=(0, 255, 0),
    ),
    Template(
        image_path="compare/beach_screen.png",
        label="3",
        color=(0, 191, 255),
        matching_threshold=0.8,
    ),
    Template(
        image_path="compare/wave.png",
        label="3",
        color=(0, 0, 255),
        matching_threshold=0.8,
    ),
    Template(
        image_path="compare/belly_button.png",
        label="3",
        color=(0, 0, 255),
        matching_threshold=0.7,
    ),
]


detector = Detector(templates, image="compare/full.png")
detector.detect()
detector.show_detections()
