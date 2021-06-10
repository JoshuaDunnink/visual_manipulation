from desktop_interaction import ScreenCapture
from image_recognition import ImageRecognizor


def main():
    camera = ScreenCapture()
    pic_id = camera.make_print_screen()
    print_screen = camera.get_filename(pic_id)

    matcher = ImageRecognizor()
    matcher.find_target_on_screen(".\compare\waldo.png", print_screen)
    matcher.show_target_on_screen()


if __name__ == "__main__":
    main()
