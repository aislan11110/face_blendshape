import cv2
import yarp
import sys
import time


def main():
    # Inicializar YARP
    if not yarp.Network.checkNetwork():
        print("[ERROR] YARP network not found!")
        return 1

    yarp.Network.init()

    # Criar porta de saída
    output_port = yarp.BufferedPortImageRgb()
    output_port.open("/webcam")

    # Abrir webcam (0 = câmera padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return 1

    # Configurar resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Webcam streaming on /webcam ({width}x{height})")
    print("[INFO] Press Ctrl+C to stop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Converter BGR (OpenCV) para RGB (YARP)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preparar imagem YARP
            yarp_image = output_port.prepare()
            yarp_image.resize(width, height)
            yarp_image.setExternal(frame_rgb.tobytes(), width, height)

            # Enviar
            output_port.write()
            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    finally:
        cap.release()
        output_port.close()
        yarp.Network.fini()

    return 0


if __name__ == '__main__':
    sys.exit(main())