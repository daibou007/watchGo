from watchGo import WatchGo
import sys

def main():
    try:
        app = WatchGo()
        if len(sys.argv) > 1:
            app.test_local_image(sys.argv[1])
        else:
            app.start()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()