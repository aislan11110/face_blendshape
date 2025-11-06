import sys
import os
import numpy as np
import cv2
import yarp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False
    print("[WARNING] DeepFace not installed. Install with: pip install deepface tf-keras")


# ================= LOGGING FUNCTIONS =================
def yarpInfo(msg):
    print(f"[INFO] {msg}")


def yarpError(msg):
    print(f"\033[91m[ERROR] {msg}\033[00m")


def yarpWarning(msg):
    print(f"\033[93m[WARNING] {msg}\033[00m")


# ================= VALENCE-AROUSAL MAPPING =================
EMOTION_TO_VA = {
    'happiness': (0.8, 0.6),
    'surprise': (0.1, 0.8),
    'anger': (-0.6, 0.7),
    'fear': (-0.7, 0.8),
    'sadness': (-0.7, -0.4),
    'disgust': (-0.6, 0.2),
    'contempt': (-0.5, 0.3),
    'neutral': (0.0, 0.0)
}

FACE_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 340), (340, 346),
    (346, 347), (347, 348), (348, 349), (349, 350), (350, 451), (451, 452),
    (452, 453), (453, 464), (464, 435), (435, 410), (410, 287), (287, 273),
    (273, 335), (335, 406), (406, 313), (313, 18), (18, 83), (83, 182),
    (182, 106), (106, 43), (43, 57), (57, 186), (186, 92), (92, 165),
    (165, 167), (167, 164), (164, 393), (393, 391), (391, 322), (322, 270),
    (270, 269), (269, 267), (267, 271), (271, 272), (272, 278), (278, 279),
    (279, 280), (280, 281), (281, 282), (282, 283), (283, 284),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307),
    (307, 375), (375, 321), (321, 308), (308, 324), (324, 318), (318, 402),
    (402, 317), (317, 14), (14, 87), (87, 178), (178, 88), (88, 95),
    (95, 78), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    (1, 2), (2, 5), (5, 4), (4, 6), (6, 168), (168, 8), (8, 55), (55, 285),
    (285, 55), (55, 8), (8, 193), (193, 168), (168, 417), (417, 351),
    (351, 419), (419, 248), (248, 281), (281, 275), (275, 305), (305, 4)
]


# ================= DATA CLASSES =================
@dataclass
class AnomalyDetection:
    """Resultado da detecção de anomalia"""
    anomaly_type: str
    confidence: float
    contributing_factors: Dict[str, float]
    timestamp: float
    valence: float
    arousal: float
    arousal_derivative: float


@dataclass
class EmotionData:
    """Dados de emoção para envio via YARP"""
    emotion: str
    confidence: float
    valence: float
    arousal: float
    is_micro: bool
    timestamp: float
    top_blendshapes: Dict[str, float]


# ================= VALENCE-AROUSAL CONVERTER =================
class ValenceArousalConverter:
    """Converte emoções categóricas para espaço Valence-Arousal"""

    def __init__(self):
        self.emotion_to_va = EMOTION_TO_VA

    def convert_categorical_to_va(self, emotion_scores: Dict[str, float]) -> Tuple[float, float]:
        """Converte scores categóricos para (V, A)"""
        valence = 0.0
        arousal = 0.0
        total_weight = sum(emotion_scores.values())

        if total_weight == 0:
            return 0.0, 0.0

        for emotion, score in emotion_scores.items():
            if emotion in self.emotion_to_va:
                v, a = self.emotion_to_va[emotion]
                weight = score / total_weight
                valence += weight * v
                arousal += weight * a

        valence = np.clip(valence, -1.0, 1.0)
        arousal = np.clip(arousal, -1.0, 1.0)

        return valence, arousal

    def convert_blendshapes_to_va(self, blendshapes: Dict[str, float]) -> Tuple[float, float]:
        """Converte blendshapes diretamente para (V, A) com sensibilidade melhorada"""
        positive_bs = [
            blendshapes.get('mouthSmileLeft', 0) * 1.5,
            blendshapes.get('mouthSmileRight', 0) * 1.5,
            blendshapes.get('cheekSquintLeft', 0) * 0.8,
            blendshapes.get('cheekSquintRight', 0) * 0.8,
            blendshapes.get('eyeSquintLeft', 0) * 0.6,
            blendshapes.get('eyeSquintRight', 0) * 0.6,
        ]

        negative_bs = [
            blendshapes.get('mouthFrownLeft', 0) * 1.5,
            blendshapes.get('mouthFrownRight', 0) * 1.5,
            blendshapes.get('browDownLeft', 0) * 1.3,
            blendshapes.get('browDownRight', 0) * 1.3,
            blendshapes.get('noseSneerLeft', 0) * 1.0,
            blendshapes.get('noseSneerRight', 0) * 1.0,
            blendshapes.get('mouthPressLeft', 0) * 0.8,
            blendshapes.get('mouthPressRight', 0) * 0.8,
        ]

        valence = np.mean(positive_bs) - np.mean(negative_bs)

        high_arousal_bs = [
            blendshapes.get('eyeWideLeft', 0) * 1.5,
            blendshapes.get('eyeWideRight', 0) * 1.5,
            blendshapes.get('browInnerUp', 0) * 1.3,
            blendshapes.get('jawOpen', 0) * 1.0,
            blendshapes.get('mouthStretchLeft', 0) * 0.8,
            blendshapes.get('mouthStretchRight', 0) * 0.8,
        ]

        low_arousal_bs = [
            blendshapes.get('eyeSquintLeft', 0) * 1.2,
            blendshapes.get('eyeSquintRight', 0) * 1.2,
            blendshapes.get('mouthClose', 0) * 0.8,
        ]

        arousal = np.mean(high_arousal_bs) - np.mean(low_arousal_bs) * 0.6

        valence = np.clip(valence * 2.5, -1.0, 1.0)
        arousal = np.clip(arousal * 2.5, -1.0, 1.0)

        return valence, arousal


# ================= MATHEMATICAL ANOMALY DETECTOR =================
class MathematicalAnomalyDetector:
    """Implementa detecção de anomalias baseada em modelo matemático"""

    def __init__(self, fps=30):
        self.fps = fps
        self.arousal_buffer = deque(maxlen=int(fps * 0.5))
        self.valence_buffer = deque(maxlen=int(fps * 0.5))
        self.timestamp_buffer = deque(maxlen=int(fps * 0.5))

        self.anomaly_params = {
            'stress': {
                'alpha': -0.7,
                'beta': 0.8,
                'gamma': -0.2,
                'lambda_I': 0.9,
                'lambda_D': 0.3
            },
            'anxiety': {
                'alpha': -0.7,
                'beta': 0.9,
                'gamma': -0.3,
                'lambda_I': 0.4,
                'lambda_D': 1.0
            },
            'deep_sadness': {
                'alpha': -0.9,
                'beta': -0.4,
                'gamma': -0.4,
                'lambda_I': 0.3,
                'lambda_D': 0.1
            },
            'suppression': {
                'alpha': 0.5,
                'beta': 0.6,
                'gamma': 0.0,
                'lambda_I': 1.0,
                'lambda_D': 0.2
            }
        }

    def sigmoid(self, x):
        """Função logística"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def calculate_arousal_derivative(self) -> float:
        """Calcula dA/dt"""
        if len(self.arousal_buffer) < 2:
            return 0.0

        arousal_values = np.array(list(self.arousal_buffer))
        time_values = np.array(list(self.timestamp_buffer))

        dA = np.diff(arousal_values)
        dt = np.diff(time_values)
        dt = np.where(dt == 0, 1e-6, dt)

        derivatives = dA / dt
        arousal_derivative = np.mean(np.abs(derivatives))

        return arousal_derivative

    def calculate_intermodal_incongruence(self, face_valence: float,
                                          face_arousal: float,
                                          physio_arousal: float = None) -> float:
        """Calcula incongruência intermodal"""
        if physio_arousal is not None:
            incongruence = np.abs(face_arousal - physio_arousal)
        else:
            arousal_derivative = self.calculate_arousal_derivative()
            if face_arousal < 0 and arousal_derivative > 0.5:
                incongruence = 0.8
            elif face_valence > 0.3 and face_arousal > 0.4:
                incongruence = 0.6
            else:
                incongruence = arousal_derivative * 0.5

        return np.clip(incongruence, 0.0, 1.0)

    def F_Anom(self, valence: float, arousal: float,
               arousal_derivative: float, incongruence: float,
               anomaly_type: str) -> float:
        """Função generalizada de ativação de anomalia"""
        if anomaly_type not in self.anomaly_params:
            return 0.0

        params = self.anomaly_params[anomaly_type]

        dimensional_term = params['alpha'] * valence + params['beta'] * arousal
        bias_term = params['gamma']
        incongruence_term = params['lambda_I'] * incongruence
        dynamics_term = params['lambda_D'] * arousal_derivative

        z = dimensional_term + bias_term + incongruence_term + dynamics_term
        activation = self.sigmoid(z)

        return activation

    def detect_anomalies(self, valence: float, arousal: float,
                         physio_arousal: float = None) -> Dict[str, float]:
        """Detecta todas as anomalias"""
        current_time = time.time()
        self.arousal_buffer.append(arousal)
        self.valence_buffer.append(valence)
        self.timestamp_buffer.append(current_time)

        arousal_derivative = self.calculate_arousal_derivative()
        incongruence = self.calculate_intermodal_incongruence(
            valence, arousal, physio_arousal
        )

        anomaly_scores = {}
        for anomaly_type in self.anomaly_params.keys():
            score = self.F_Anom(
                valence, arousal, arousal_derivative,
                incongruence, anomaly_type
            )
            anomaly_scores[anomaly_type] = score

        anomaly_scores['_debug'] = {
            'arousal_derivative': arousal_derivative,
            'incongruence': incongruence,
            'valence': valence,
            'arousal': arousal
        }

        return anomaly_scores

    def get_dominant_anomaly(self, anomaly_scores: Dict[str, float],
                             threshold: float = 0.55) -> Optional[AnomalyDetection]:
        """Retorna anomalia dominante se ultrapassar threshold"""
        scores = {k: v for k, v in anomaly_scores.items() if k != '_debug'}

        if not scores:
            return None

        max_anomaly = max(scores, key=scores.get)
        max_score = scores[max_anomaly]

        if max_score < threshold:
            return None

        debug = anomaly_scores.get('_debug', {})

        return AnomalyDetection(
            anomaly_type=max_anomaly,
            confidence=max_score,
            contributing_factors=scores,
            timestamp=time.time(),
            valence=debug.get('valence', 0),
            arousal=debug.get('arousal', 0),
            arousal_derivative=debug.get('arousal_derivative', 0)
        )


# ================= YARP MODULE PRINCIPAL =================
class FaceBlendshapesModule(yarp.RFModule):
    """
    YARP Module: Face Blendshapes + Valence-Arousal + Anomaly Detection

    Ports:
        /faceBlendshapes/image:i - Entrada RGB
        /faceBlendshapes/annotated_image:o - Imagem com landmarks
        /faceBlendshapes/blendshapes:o - Blendshapes (52)
        /faceBlendshapes/emotion:o - Emoção + V-A
        /faceBlendshapes/anomaly:o - Detecção de anomalias
        /faceBlendshapes/rpc - Comandos RPC
    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        self.detector = None
        self.frame = None

        # Handle port for RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Input/Output image ports
        self.input_port = yarp.BufferedPortImageRgb()
        self.output_img_port = yarp.BufferedPortImageRgb()

        # Data output ports
        self.output_blendshapes_port = yarp.Port()
        self.output_emotion_port = yarp.Port()
        self.output_anomaly_port = yarp.Port()

        # Image buffers
        self.input_img_array = None
        self.display_buf_image = yarp.ImageRgb()
        self.width_img = 640
        self.height_img = 480

        # Processing flags
        self.process = True
        self.draw_landmarks_flag = True
        self.top_n_blendshapes = 10
        self.send_emotion = True
        self.send_anomalies = True

        # Converters and detectors
        self.va_converter = ValenceArousalConverter()
        self.anomaly_detector = MathematicalAnomalyDetector(fps=30)

        # Baseline e estado
        self.baseline = None
        self.baseline_frames = deque(maxlen=90)
        self.blendshape_buffer = deque(maxlen=15)
        self.emotion_history = deque(maxlen=20)

        # Detecção de expressões
        self.expression_start_time = None
        self.expression_start_blendshapes = None
        self.peak_intensity = 0

        # Detectar mudanças
        self.last_blendshapes = None
        self.expression_change_threshold = 0.18

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        """Configuração do módulo"""

        if rf.check('help') or rf.check('h'):
            print("Face Blendshapes Module (with V-A + Anomalies):")
            print("\t--name (default faceBlendshapes) module name")
            print("\t--model_path (default ./face_landmarker.task) MediaPipe model path")
            print("\t--draw_landmarks (default True) draw landmarks")
            print("\t--top_n (default 10) top blendshapes to output")
            print("\t--send_emotion (default True) send emotion data")
            print("\t--send_anomalies (default True) send anomaly detection")
            return False

        self.module_name = rf.check("name",
                                    yarp.Value("faceBlendshapes"),
                                    "module name").asString()

        self.model_path = rf.check("model_path",
                                   yarp.Value("face_landmarker.task"),
                                   "path to MediaPipe model").asString()

        self.draw_landmarks_flag = rf.check("draw_landmarks",
                                            yarp.Value(True),
                                            "draw landmarks").asBool()

        self.top_n_blendshapes = rf.check("top_n",
                                          yarp.Value(10),
                                          "top N blendshapes").asInt32()

        self.send_emotion = rf.check("send_emotion",
                                     yarp.Value(True),
                                     "send emotion data").asBool()

        self.send_anomalies = rf.check("send_anomalies",
                                       yarp.Value(True),
                                       "send anomaly detection").asBool()

        # Initialize MediaPipe
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

        if self.send_emotion:
            self.output_emotion_port.open('/' + self.module_name + '/emotion:o')

        if self.send_anomalies:
            self.output_anomaly_port.open('/' + self.module_name + '/anomaly:o')

        # Initialize image arrays
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.display_buf_image.resize(self.width_img, self.height_img)

        yarpInfo('Module initialized successfully')
        yarpInfo(f'Emotion output: {self.send_emotion}')
        yarpInfo(f'Anomaly detection: {self.send_anomalies}')

        return True

    def get_image_from_yarp(self, yarp_image):
        """Converte imagem YARP para numpy array"""
        if yarp_image.width() != self.width_img or yarp_image.height() != self.height_img:
            yarpWarning("Input image size changed, adapting...")
            self.width_img = yarp_image.width()
            self.height_img = yarp_image.height()
            self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

        yarp_image.setExternal(self.input_img_array.data, self.width_img, self.height_img)
        frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.height_img, self.width_img, 3))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def draw_landmarks_on_image(self, image, face_landmarks):
        """Desenha landmarks faciais"""
        height, width = image.shape[:2]

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

        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    def update_baseline(self, blendshapes: Dict[str, float]):
        """Atualiza baseline com média móvel"""
        self.baseline_frames.append(blendshapes)

        if len(self.baseline_frames) > 30:
            self.baseline = {}
            for key in blendshapes.keys():
                values = [frame[key] for frame in self.baseline_frames]
                self.baseline[key] = np.mean(values)

    def detect_expression_change(self, blendshapes: Dict[str, float]) -> bool:
        """Detecta mudança significativa de expressão - VERSÃO SENSÍVEL"""
        if self.baseline is None:
            return False

        total_change = 0
        significant_changes = 0

        for key, value in blendshapes.items():
            baseline_val = self.baseline.get(key, 0)
            change = abs(value - baseline_val)
            total_change += change

            if change > 0.1:
                significant_changes += 1

        threshold = 0.18 * len(blendshapes) / 52

        has_change = total_change > threshold or significant_changes >= 3

        return has_change

    def blendshapes_to_bottle(self, blendshapes_dict: Dict[str, float]) -> yarp.Bottle:
        """Converte blendshapes para YARP bottle"""
        bottle = yarp.Bottle()

        sorted_blendshapes = sorted(blendshapes_dict.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:self.top_n_blendshapes]

        for name, value in sorted_blendshapes:
            bs_bottle = yarp.Bottle()
            bs_bottle.addString(name)
            bs_bottle.addFloat64(float(value))
            bottle.addList().read(bs_bottle)

        return bottle

    def emotion_to_bottle(self, emotion: str, confidence: float,
                          valence: float, arousal: float) -> yarp.Bottle:
        """Cria YARP bottle com dados de emoção"""
        bottle = yarp.Bottle()
        bottle.addString(emotion)
        bottle.addFloat64(confidence)
        bottle.addFloat64(valence)
        bottle.addFloat64(arousal)
        return bottle

    def anomaly_to_bottle(self, anomaly: AnomalyDetection) -> yarp.Bottle:
        """Cria YARP bottle com dados de anomalia"""
        bottle = yarp.Bottle()
        bottle.addString(anomaly.anomaly_type)
        bottle.addFloat64(anomaly.confidence)
        bottle.addFloat64(anomaly.valence)
        bottle.addFloat64(anomaly.arousal)
        bottle.addFloat64(anomaly.arousal_derivative)

        # Adicionar fatores contribuintes
        factors = yarp.Bottle()
        for factor_name, factor_value in anomaly.contributing_factors.items():
            if factor_name != '_debug':
                f_bottle = yarp.Bottle()
                f_bottle.addString(factor_name)
                f_bottle.addFloat64(factor_value)
                factors.addList().read(f_bottle)
        bottle.addList().read(factors)

        return bottle

    def updateModule(self):
        """Update do módulo (executado periodicamente)"""
        if not self.process:
            return True

        # Ler imagem da porta
        message = self.input_port.read(False)

        if message is not None:
            self.frame = self.get_image_from_yarp(message)

            # Processar com MediaPipe
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = self.detector.detect(mp_image)

            annotated_frame = self.frame.copy()

            if result.face_landmarks and len(result.face_landmarks) > 0:
                face_landmarks = result.face_landmarks[0]

                if self.draw_landmarks_flag:
                    self.draw_landmarks_on_image(annotated_frame, face_landmarks)

                if result.face_blendshapes and len(result.face_blendshapes) > 0:
                    blendshapes_dict = {b.category_name: b.score
                                        for b in result.face_blendshapes[0]}

                    # Atualizar baseline
                    if self.baseline is None:
                        self.update_baseline(blendshapes_dict)
                        return True

                    self.blendshape_buffer.append(blendshapes_dict)

                    # Enviar blendshapes
                    if self.output_blendshapes_port.getOutputCount():
                        bottle = self.blendshapes_to_bottle(blendshapes_dict)
                        self.output_blendshapes_port.write(bottle)

                    # ============= PROCESSAMENTO DE EMOÇÃO E ANOMALIAS =============
                    if self.send_emotion or self.send_anomalies:
                        # Calcular V-A
                        valence_cat, arousal_cat = self.va_converter.convert_categorical_to_va(
                            {'neutral': 1.0}  # Placeholder para regras simples
                        )
                        valence_bs, arousal_bs = self.va_converter.convert_blendshapes_to_va(blendshapes_dict)

                        valence = 0.5 * valence_cat + 0.5 * valence_bs
                        arousal = 0.5 * arousal_cat + 0.5 * arousal_bs

                        # Detectar anomalias
                        if self.send_anomalies:
                            anomaly_scores = self.anomaly_detector.detect_anomalies(valence, arousal)
                            detected_anomaly = self.anomaly_detector.get_dominant_anomaly(
                                anomaly_scores,
                                threshold=0.55
                            )

                            if detected_anomaly and self.output_anomaly_port.getOutputCount():
                                bottle = self.anomaly_to_bottle(detected_anomaly)
                                self.output_anomaly_port.write(bottle)

                        # Enviar emoção
                        if self.send_emotion and self.output_emotion_port.getOutputCount():
                            max_blend = max(blendshapes_dict.values()) if blendshapes_dict else 0
                            emotion = "neutral"  # Placeholder
                            confidence = max_blend

                            bottle = self.emotion_to_bottle(emotion, confidence, valence, arousal)
                            self.output_emotion_port.write(bottle)

            # Enviar imagem anotada
            if self.output_img_port.getOutputCount():
                self.write_annotated_image(annotated_frame)

        return True

    def write_annotated_image(self, annotated_image):
        """Envia imagem anotada na porta de saída"""
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        self.display_buf_image = self.output_img_port.prepare()
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(annotated_image.tobytes(),
                                           self.width_img, self.height_img)
        self.output_img_port.write()

    def respond(self, command, reply):
        """Responde a comandos RPC"""
        reply.clear()

        try:
            cmd = command.get(0).asString() if command.size() > 0 else ""

            if cmd == "quit":
                reply.addString("quitting")
                return False

            elif cmd == "help":
                help_text = (
                    "Face Blendshapes Module Commands:\n"
                    "  process [on/off] - Enable/disable processing\n"
                    "  landmarks [on/off] - Toggle landmark drawing\n"
                    "  emotion [on/off] - Toggle emotion output\n"
                    "  anomalies [on/off] - Toggle anomaly detection\n"
                    "  status - Get module status\n"
                    "  quit - Quit module\n"
                )
                reply.addString(help_text)

            elif cmd == "process":
                if command.size() > 1:
                    self.process = command.get(1).asString() == 'on'
                    reply.addString("ok")
                else:
                    reply.addString("usage: process [on/off]")

            elif cmd == "landmarks":
                if command.size() > 1:
                    self.draw_landmarks_flag = command.get(1).asString() == 'on'
                    reply.addString("ok")
                else:
                    reply.addString("usage: landmarks [on/off]")

            elif cmd == "emotion":
                if command.size() > 1:
                    self.send_emotion = command.get(1).asString() == 'on'
                    reply.addString("ok")
                else:
                    reply.addString("usage: emotion [on/off]")

            elif cmd == "anomalies":
                if command.size() > 1:
                    self.send_anomalies = command.get(1).asString() == 'on'
                    reply.addString("ok")
                else:
                    reply.addString("usage: anomalies [on/off]")

            elif cmd == "status":
                status = (
                    f"Processing: {self.process}\n"
                    f"Landmarks: {self.draw_landmarks_flag}\n"
                    f"Emotion: {self.send_emotion}\n"
                    f"Anomalies: {self.send_anomalies}\n"
                    f"Frame size: {self.width_img}x{self.height_img}\n"
                    f"Baseline: {'Yes' if self.baseline else 'No'}\n"
                )
                reply.addString(status)

            else:
                reply.addString("unknown command (type 'help' for usage)")

        except Exception as e:
            yarpError(f"Error in respond: {e}")
            reply.addString("error")

        return True

    def getPeriod(self):
        """Período de atualização do módulo (segundos)"""
        return 0.01

    def interruptModule(self):
        """Interrompe o módulo"""
        yarpInfo("Stopping module")
        self.handle_port.interrupt()
        self.input_port.interrupt()
        self.output_img_port.interrupt()
        self.output_blendshapes_port.interrupt()
        if self.send_emotion:
            self.output_emotion_port.interrupt()
        if self.send_anomalies:
            self.output_anomaly_port.interrupt()
        return True

    def close(self):
        """Fecha o módulo"""
        yarpInfo("Closing module")
        self.handle_port.close()
        self.input_port.close()
        self.output_img_port.close()
        self.output_blendshapes_port.close()
        if self.send_emotion:
            self.output_emotion_port.close()
        if self.send_anomalies:
            self.output_anomaly_port.close()
        return True


if __name__ == '__main__':
    # Initialize YARP
    if not yarp.Network.checkNetwork():
        yarpError("YARP network not found!")
        sys.exit(1)

    yarp.Network.init()

    module = FaceBlendshapesModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('faceBlendshapes')
    rf.setDefaultConfigFile('faceBlendshapes.ini')

    if rf.configure(sys.argv):
        yarpInfo("Module running...")
        module.runModule(rf)

    sys.exit()