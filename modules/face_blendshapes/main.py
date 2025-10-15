import sys
import os
import numpy as np
import cv2
import yarp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def yarpInfo(msg):
    print("[INFO] {}".format(msg))


def yarpError(msg):
    print("\033[91m[ERROR] {}\033[00m".format(msg))


def yarpWarning(msg):
    print("\033[93m[WARNING] {}\033[00m".format(msg))


# Conexões dos landmarks faciais
FACE_CONNECTIONS = [
    # Contorno do rosto
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 340), (340, 346),
    (346, 347), (347, 348), (348, 349), (349, 350), (350, 451), (451, 452),
    (452, 453), (453, 464), (464, 435), (435, 410), (410, 287), (287, 273),
    (273, 335), (335, 406), (406, 313), (313, 18), (18, 83), (83, 182),
    (182, 106), (106, 43), (43, 57), (57, 186), (186, 92), (92, 165),
    (165, 167), (167, 164), (164, 393), (393, 391), (391, 322), (322, 270),
    (270, 269), (269, 267), (267, 271), (271, 272), (272, 278), (278, 279),
    (279, 280), (280, 281), (281, 282), (282, 283), (283, 284),
    # Olho esquerdo
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    # Olho direito
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    # Lábios externos
    (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307),
    (307, 375), (375, 321), (321, 308), (308, 324), (324, 318), (318, 402),
    (402, 317), (317, 14), (14, 87), (87, 178), (178, 88), (88, 95),
    (95, 78), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Nariz
    (1, 2), (2, 5), (5, 4), (4, 6), (6, 168), (168, 8), (8, 55), (55, 285),
    (285, 55), (55, 8), (8, 193), (193, 168), (168, 417), (417, 351),
    (351, 419), (419, 248), (248, 281), (281, 275), (275, 305), (305, 4)
]


class FaceBlendshapesModule(yarp.RFModule):
    """
    YARP Module for face landmark detection and blendshapes extraction using MediaPipe

    Ports:
        input_port: receives RGB images
        output_img_port: sends annotated images with landmarks
        output_blendshapes_port: sends blendshapes data as YARP bottles
    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        self.detector = None
        self.frame = None

        # Handle port for RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Input image port
        self.input_port = yarp.BufferedPortImageRgb()
        self.input_img_array = None
        self.width_img = 640
        self.height_img = 480

        # Output annotated image port
        self.output_img_port = yarp.BufferedPortImageRgb()
        self.display_buf_array = None
        self.display_buf_image = yarp.ImageRgb()

        # Output blendshapes port
        self.output_blendshapes_port = yarp.Port()

        # Processing flag
        self.process = True
        self.draw_landmarks_flag = True
        self.top_n_blendshapes = 10

    def configure(self, rf: yarp.ResourceFinder) -> bool:

        if rf.check('help') or rf.check('h'):
            print("Face Blendshapes Module options:")
            print("\t--name (default faceBlendshapes) module name")
            print("\t--model_path (default ./face_landmarker.task) path to MediaPipe model")
            print("\t--draw_landmarks (default True) draw landmarks on output image")
            print("\t--top_n (default 10) number of top blendshapes to output")
            print("\t--help print this help")
            return False

        self.module_name = rf.check("name",
                                    yarp.Value("faceBlendshapes"),
                                    "module name (string)").asString()

        self.model_path = rf.check("model_path",
                                   yarp.Value("face_landmarker.task"),
                                   "path to MediaPipe model").asString()

        self.draw_landmarks_flag = rf.check("draw_landmarks",
                                            yarp.Value(True),
                                            "draw landmarks on output").asBool()

        self.top_n_blendshapes = rf.check("top_n",
                                          yarp.Value(10),
                                          "number of top blendshapes").asInt32()

        # Initialize MediaPipe detector
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            yarpInfo("MediaPipe Face Landmarker loaded successfully")
        except Exception as e:
            yarpError(f"Failed to load MediaPipe model: {e}")
            return False

        # Open ports
        self.handle_port.open('/' + self.module_name)
        self.input_port.open('/' + self.module_name + '/image:i')
        self.output_img_port.open('/' + self.module_name + '/annotated_image:o')
        self.output_blendshapes_port.open('/' + self.module_name + '/blendshapes:o')

        # Initialize image arrays
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.display_buf_image.setExternal(self.display_buf_array.data, self.width_img, self.height_img)

        yarpInfo('Module initialized successfully')
        return True

    def get_image_from_yarp(self, yarp_image):
        """Convert YARP image to numpy array"""
        if yarp_image.width() != self.width_img or yarp_image.height() != self.height_img:
            yarpWarning("Input image size changed, adapting...")
            self.width_img = yarp_image.width()
            self.height_img = yarp_image.height()
            self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

        yarp_image.setExternal(self.input_img_array.data, self.width_img, self.height_img)
        frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.height_img, self.width_img, 3))

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def draw_landmarks_on_image(self, image, face_landmarks):
        """Draw facial landmarks on the image"""
        height, width = image.shape[:2]

        # Draw connections
        for connection in FACE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                start_point = face_landmarks[start_idx]
                end_point = face_landmarks[end_idx]

                x1 = int(start_point.x * width)
                y1 = int(start_point.y * height)
                x2 = int(end_point.x * width)
                y2 = int(end_point.y * height)

                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Draw points
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    def blendshapes_to_bottle(self, blendshapes_dict):
        """Convert blendshapes dictionary to YARP bottle"""
        bottle = yarp.Bottle()

        # Sort and get top N
        sorted_blendshapes = sorted(blendshapes_dict.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:self.top_n_blendshapes]

        for name, value in sorted_blendshapes:
            bs_bottle = yarp.Bottle()
            bs_bottle.addString(name)
            bs_bottle.addFloat64(float(value))
            bottle.addList().read(bs_bottle)

        return bottle

    def updateModule(self):
        if not self.process:
            return True

        # Read image from port
        message = self.input_port.read(False)

        if message is not None:
            # Convert YARP image to numpy
            self.frame = self.get_image_from_yarp(message)

            # Convert to MediaPipe format
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face + blendshapes
            result = self.detector.detect(mp_image)

            annotated_frame = self.frame.copy()

            # Process results
            if result.face_landmarks and len(result.face_landmarks) > 0:
                face_landmarks = result.face_landmarks[0]

                # Draw landmarks if enabled
                if self.draw_landmarks_flag:
                    self.draw_landmarks_on_image(annotated_frame, face_landmarks)

                # Process blendshapes
                if result.face_blendshapes and len(result.face_blendshapes) > 0:
                    blendshapes_dict = {b.category_name: b.score
                                        for b in result.face_blendshapes[0]}

                    # Send blendshapes via YARP if port is connected
                    if self.output_blendshapes_port.getOutputCount():
                        bottle = self.blendshapes_to_bottle(blendshapes_dict)
                        self.output_blendshapes_port.write(bottle)

            # Send annotated image if port is connected
            if self.output_img_port.getOutputCount():
                self.write_annotated_image(annotated_frame)

        return True

    def write_annotated_image(self, annotated_image):
        """Stream annotated image on YARP port"""
        # Convert BGR to RGB for YARP
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        self.display_buf_image = self.output_img_port.prepare()
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(annotated_image.tobytes(),
                                           self.width_img, self.height_img)
        self.output_img_port.write()

    def respond(self, command, reply):
        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "help":
            reply.addString("Face Blendshapes module commands:\n")
            reply.addString("process on/off -> enable/disable processing\n")
            reply.addString("landmarks on/off -> enable/disable landmark drawing\n")
            reply.addString("quit -> quit the module\n")

        elif command.get(0).asString() == "process":
            self.process = True if command.get(1).asString() == 'on' else False
            reply.addString("ok")

        elif command.get(0).asString() == "landmarks":
            self.draw_landmarks_flag = True if command.get(1).asString() == 'on' else False
            reply.addString("ok")

        else:
            reply.addString("unknown command")

        return True

    def getPeriod(self):
        """Module refresh rate in seconds"""
        return 0.01

    def interruptModule(self):
        yarpInfo("Stopping the module")
        self.handle_port.interrupt()
        self.input_port.interrupt()
        self.output_img_port.interrupt()
        self.output_blendshapes_port.interrupt()
        return True

    def close(self):
        yarpInfo("Closing the module")
        self.handle_port.close()
        self.input_port.close()
        self.output_img_port.close()
        self.output_blendshapes_port.close()
        return True


if __name__ == '__main__':
    # Initialize YARP
    if not yarp.Network.checkNetwork():
        yarpError("Unable to find a yarp server, exiting...")
        sys.exit(1)

    yarp.Network.init()

    module = FaceBlendshapesModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('faceBlendshapes')
    rf.setDefaultConfigFile('faceBlendshapes.ini')

    if rf.configure(sys.argv):
        module.runModule(rf)

    sys.exit()