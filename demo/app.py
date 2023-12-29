import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

from predict import *


def main():
    # 카메라 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

    # Streamlit 앱 레이아웃 설정
    st.sidebar.header("설정")
    frame_rate = st.sidebar.slider("프레임 속도", 1, 30, 15)

    # Streamlit 앱 시작
    st.write("## 공사장 안전 장구 착용 여부 모니터링 서비스")
    st.write("#### 탐지 안전 장구: 안전모, 안전화, 안전벨트")

    # 결과 이미지를 업데이트할 컨테이너 생성
    result_container = st.empty()

    # 결과 내용을 업데이트할 컨테이너 생성
    result_text = st.empty()

    # OpenCV의 카메라 캡쳐 루프
    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            st.warning("카메라에서 프레임을 읽을 수 없습니다!")
            break

        # OpenCV 프레임을 JPEG 이미지로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        # 객체 감지 예측 수행
        results = predict(frame)

        # 결과 시각화
        result_image = frame.copy()
        df_results = pd.DataFrame(results.pandas().xyxy[0])
        for index, row in df_results.iterrows():
            label = row['name']
            confidence = row["confidence"]
            x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
            colors = [(255, 0, 0), (0, 255, 0)]  # 바운딩 박스 색상
            thickness = 5  # 바운딩 박스 두께
            font_scale = 1.0  # 바운딩 박스 레이블 글꼴 크기
            font_thickness = 2  # 바운딩 박스 레이블 글꼴 두께

            if row["class"] % 2 == 1:
                color = colors[0]
            else:
                color = colors[1]
            # OpenCV를 사용하여 바운딩 박스 그리기
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(result_image, f"{label}: {confidence:.2f}", (
                x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        result_container.image(result_image, caption='Real-time Detection',
                               use_column_width=True)

        # 결과 내용 출력
        df_results = pd.DataFrame(results.pandas().xyxy[0])
        results_text = "<h3>미착용 탐지시 출력</h3>"
        for index, row in df_results.iterrows():
            if row["class"] % 2 == 1:
                results_text += f"<div><span style='font-size: 20px;'>{row['name']} / </span><span>Confidence: {row['confidence']:.2f}</span></div>"

        # HTML 태그를 사용하여 줄바꿈을 추가
        result_text.markdown(results_text, unsafe_allow_html=True)

        # 프레임 속도 조절
        st.experimental_set_query_params(fps=frame_rate)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
