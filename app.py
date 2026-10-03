# -*- coding: utf-8 -*-
"""Demo phân loại hoa Iris bằng Gaussian Naive Bayes - giao diện Streamlit."""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# ---------- 1. Cấu hình trang ----------
st.set_page_config(page_title="Demo Phân Loại Hoa Iris", page_icon="🌸", layout="wide")

# Thư mục chứa ảnh
THU_MUC_ANH = [
    "images",                            
    "/content/drive/MyDrive/images",     
]
# Ảnh dự phòng (cần Internet) nếu không tìm thấy ảnh trong thư mục trên
ANH_DU_PHONG = {
    "setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
}

# ---------- 2. Huấn luyện mô hình (chỉ chạy 1 lần nhờ cache) ----------
@st.cache_resource
def tai_mo_hinh():
    """Tải Iris, huấn luyện GaussianNB trên toàn bộ dữ liệu."""
    iris = load_iris()
    model = GaussianNB()
    model.fit(iris.data, iris.target)
    return model, list(iris.target_names)

def du_doan(model, sl, sw, pl, pw):
    """Trả về (chỉ số lớp dự đoán, mảng xác suất 3 lớp)."""
    x = np.array([[sl, sw, pl, pw]])
    proba = model.predict_proba(x)[0]
    return int(np.argmax(proba)), proba

def lay_anh(ten_loai):
    """Tìm file ảnh có chứa tên loài (không phân biệt hoa/thường, vd 'Iris setosa.jpg')."""
    for thu_muc in THU_MUC_ANH:
        if os.path.isdir(thu_muc):
            for ten_file in sorted(os.listdir(thu_muc)):
                if ten_loai in ten_file.lower() and ten_file.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")):
                    return os.path.join(thu_muc, ten_file)
    return ANH_DU_PHONG[ten_loai]  # không thấy ảnh cục bộ -> dùng ảnh trên mạng

# ---------- 3. Tiêu đề ----------
model, ten_loai = tai_mo_hinh()

st.title(" Demo Phân Loại Hoa Iris Dataset")
st.subheader("Mô hình: Gaussian Naive Bayes")
st.markdown("**Hướng dẫn:** kéo 4 thanh trượt để nhập kích thước hoa, "
            "sau đó bấm **Dự đoán ngay** để xem kết quả.")
st.divider()

# ---------- 4. Bố cục 2 cột ----------
cot_trai, cot_phai = st.columns(2)

with cot_trai:
    st.header(" Đầu vào")
    sl = st.slider("Chiều dài lá đài (Sepal Length - cm)", 4.0, 8.0, 5.1, 0.1)
    sw = st.slider("Chiều rộng lá đài (Sepal Width - cm)", 2.0, 4.5, 3.5, 0.1)
    pl = st.slider("Chiều dài cánh hoa (Petal Length - cm)", 1.0, 7.0, 1.4, 0.1)
    pw = st.slider("Chiều rộng cánh hoa (Petal Width - cm)", 0.1, 2.5, 0.2, 0.1)
    bam_nut = st.button("Dự đoán ngay", type="primary")

with cot_phai:
    st.header(" Kết quả")
    # Chỉ hiển thị kết quả sau khi bấm nút
    if bam_nut:
        idx, proba = du_doan(model, sl, sw, pl, pw)
        loai = ten_loai[idx]

        # Tên loài: in hoa, in đậm
        st.markdown(f"### Loài dự đoán: **{loai.upper()}**")

        # Hình ảnh minh họa
        st.image(lay_anh(loai), caption=f"Iris {loai}", width=350)

        # Xác suất của cả 3 loài: thanh tiến trình + biểu đồ cột
        st.markdown("**Xác suất từng loài:**")
        for ten, p in zip(ten_loai, proba):
            st.progress(float(p), text=f"{ten}: {p*100:.2f}%")
        st.bar_chart(pd.DataFrame({"Xác suất": proba}, index=ten_loai))
    else:
        st.info("Chưa có kết quả. Hãy chọn giá trị và bấm **Dự đoán ngay**.")
