import os
import time
from uuid import uuid4

import pyautogui


class ScreenCapture:
    def __init__(self, storage_path="screencaptures"):
        self.storage_path = storage_path
        self.storage_list = {}

    def make_print_screen(self):
        screenshot = pyautogui.screenshot()

        uuid = str(uuid4())
        filename = os.path.join(self.storage_path, (uuid + ".png"))

        screenshot.save(filename)

        self.storage_list.update(
            {
                uuid: {
                    "timestamp": time.time(),
                    "filename": filename,
                }
            }
        )
        return uuid

    def get_filename(self, uuid):
        return self.storage_list.get(uuid).get("filename")

    def remove_print_screen(self, id):
        file = self.storage_list.get(id)
        self.storage_list.pop(id)
        os.remove(file.get("filename"))


class ScreenClicker:
    def __init__(self):
        pass

    def click_at(self, x, y):
        pyautogui.click(x, y)
