import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Avaliação de Avaria Veicular", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def calculate_damage_score(image, box=None):
    """
    Calcula a gravidade do dano com base na variação estrutural de textura,
    contraste e ruído de deformação metalúrgica.
    """
    if box is not None:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image
    else:
        roi = image

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Variância do Laplaciano: Mede descontinuidade e deformação metálica brusca
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Detecção de bordas desordenadas (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / (roi.shape[0] * roi.shape[1])
    
    # Métrica combinada de deformação
    combined_score = (laplacian_var / 1000.0) + (edge_ratio * 10.0)
    return combined_score

st.title("Sistema de Avaliação de Avaria Veicular")
st.write("Faça upload da imagem para identificar o nível de dano do veículo.")

uploaded_file = st.file_uploader("Selecione uma imagem (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # OTIMIZAÇÃO DE MEMÓRIA: Redimensiona imagens grandes para evitar estouro de RAM
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # OTIMIZAÇÃO DE MEMÓRIA: Limita a resolução interna do YOLO em 640px
    results = model.predict(source=image, device="cpu", imgsz=640, conf=0.25, verbose=False)
    
    annotated_img = image.copy()
    max_score = 0.0
    detected_vehicles = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()
            
            if class_name in ["car", "truck", "bus"]:
                detected_vehicles += 1
                score = calculate_damage_score(image, box)
                if score > max_score:
                    max_score = score
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Fallback: Se o YOLO não identificar a silhueta inteira (ex: foto cortada do dano)
    if detected_vehicles == 0 or max_score == 0:
        max_score = calculate_damage_score(image, None)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img_rgb, caption="Imagem Processada", use_container_width=True)

    st.subheader("Resultado da Análise:")

    # Calibração dos níveis de avaria
    if max_score > 3.8:
        st.error("**Status:** BATIDA PT (AVARIA 3)\n\n**Ação:** Encaminhar para Perda Total")
    elif max_score > 2.2:
        st.warning("**Status:** BATIDA MÉDIA (AVARIA 2)\n\n**Ação:** Encaminhar para Mecânica")
    elif max_score > 1.3:
        st.info("**Status:** BATIDA LEVE (AVARIA 1)\n\n**Ação:** Encaminhar para Funilaria")
    else:
        st.success("**Status:** SEM AVARIAS\n\n**Ação:** Veículo em perfeito estado ou sem danos visíveis.")
