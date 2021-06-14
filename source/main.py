from desktop_interaction import ScreenCapture
from image_recognition import ImageRecognizor
from timer import timer

@timer
def main():
    camera = ScreenCapture()
    pic_id = camera.make_print_screen()
    print_screen = camera.get_filename(pic_id)

    matcher = ImageRecognizor()
    matcher.find_target_on_screen(".\\compare\\waldo.png", print_screen)
    matcher.show_target_on_screen(wait=1)


if __name__ == "__main__":
    for _ in range(10):
        main()
